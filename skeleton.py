# -*- coding: utf-8 -*-

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from lib.models.vibe import VIBE_Demo
from lib.utils.renderer import Renderer
from lib.dataset.inference import Inference
from lib.utils.smooth_pose import smooth_pose
from lib.data_utils.kp_utils import convert_kps
from lib.utils.pose_tracker import run_posetracker

from lib.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_coords_to_orig_img,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)
import shutil

MIN_NUM_FRAMES = 25


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    video_file = args.vid_file
    outFileName = os.path.join(args.output_folder, os.path.basename(video_file).replace('_rgb.avi', '.skeleton'))
    if os.path.isfile(outFileName):
        print(outFileName+" exists!")
        return 0
    '''
    # ========= [Optional] download the youtube video ========= #
    if video_file.startswith('https://www.youtube.com'):
        print(f'Donwloading YouTube video \"{video_file}\"')
        video_file = download_youtube_clip(video_file, '/tmp')

        if video_file is None:
            exit('Youtube url is not valid!')

        print(f'YouTube Video has been downloaded to {video_file}...')
    '''
    if not os.path.isfile(video_file):
        exit(f'Input video \"{video_file}\" does not exist!')

    #output_path = os.path.join(args.output_folder, os.path.basename(video_file).replace('.mp4', ''))
    #os.makedirs(output_path, exist_ok=True)

    image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)

    print(f'Input video number of frames {num_frames}')
    orig_height, orig_width = img_shape[:2]

    total_time = time.time()

    # ========= Run tracking ========= #
    bbox_scale = 1.1
    if args.tracking_method == 'pose':
        if not os.path.isabs(video_file):
            video_file = os.path.join(os.getcwd(), video_file)
        tracking_results = run_posetracker(video_file, staf_folder=args.staf_dir, display=args.display)
    else:
        # run multi object tracker
        mot = MPT(
            device=device,
            batch_size=args.tracker_batch_size,
            display=args.display,
            detector_type=args.detector,
            output_format='dict',
            yolo_img_size=args.yolo_img_size,
        )
        tracking_results = mot(image_folder)

    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]

    # ========= Define VIBE model ========= #
    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)

    # ========= Load pretrained weights ========= #
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')

    # ========= Run VIBE on each person ========= #
    print(f'Running VIBE on each tracklet...')
    vibe_time = time.time()
    vibe_results = {}

    # Write skeletal data
    outFileName = os.path.join(args.output_folder, os.path.basename(video_file).replace('_rgb.avi', '.skeleton'))
    outFile = open(outFileName, "w")
    #outFile.write(str(num_frames)+"\n")
    
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = joints2d = None

        if args.tracking_method == 'bbox':
            bboxes = tracking_results[person_id]['bbox']
        elif args.tracking_method == 'pose':
            joints2d = tracking_results[person_id]['joints2d']

        frames = tracking_results[person_id]['frames']

        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=bbox_scale,
        )

        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if joints2d is not None else False

        dataloader = DataLoader(dataset, batch_size=args.vibe_batch_size, num_workers=16)

        with torch.no_grad():

            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, smpl_joints2d, norm_joints2d = [], [], [], [], [], [], []

            for batch in dataloader:
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.unsqueeze(0)
                batch = batch.to(device)

                batch_size, seqlen = batch.shape[:2]
                output = model(batch)[-1]

                pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1))
                pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1))
                pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))
                smpl_joints2d.append(output['kp_2d'].reshape(batch_size * seqlen, -1, 2))


            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)
            smpl_joints2d = torch.cat(smpl_joints2d, dim=0)
            del batch

        # ========= [Optional] run Temporal SMPLify to refine the results ========= #
        if args.run_smplify and args.tracking_method == 'pose':
            norm_joints2d = np.concatenate(norm_joints2d, axis=0)
            norm_joints2d = convert_kps(norm_joints2d, src='staf', dst='spin')
            norm_joints2d = torch.from_numpy(norm_joints2d).float().to(device)

            # Run Temporal SMPLify
            update, new_opt_vertices, new_opt_cam, new_opt_pose, new_opt_betas, \
            new_opt_joints3d, new_opt_joint_loss, opt_joint_loss = smplify_runner(
                pred_rotmat=pred_pose,
                pred_betas=pred_betas,
                pred_cam=pred_cam,
                j2d=norm_joints2d,
                device=device,
                batch_size=norm_joints2d.shape[0],
                pose2aa=False,
            )

            # update the parameters after refinement
            print(f'Update ratio after Temporal SMPLify: {update.sum()} / {norm_joints2d.shape[0]}')
            pred_verts = pred_verts.cpu()
            pred_cam = pred_cam.cpu()
            pred_pose = pred_pose.cpu()
            pred_betas = pred_betas.cpu()
            pred_joints3d = pred_joints3d.cpu()
            pred_verts[update] = new_opt_vertices[update]
            pred_cam[update] = new_opt_cam[update]
            pred_pose[update] = new_opt_pose[update]
            pred_betas[update] = new_opt_betas[update]
            pred_joints3d[update] = new_opt_joints3d[update]

        elif args.run_smplify and args.tracking_method == 'bbox':
            print('[WARNING] You need to enable pose tracking to run Temporal SMPLify algorithm!')
            print('[WARNING] Continuing without running Temporal SMPLify!..')

        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()
        smpl_joints2d = smpl_joints2d.cpu().numpy()

        # Runs 1 Euro Filter to smooth out the results
        if args.smooth:
            min_cutoff = args.smooth_min_cutoff # 0.004
            beta = args.smooth_beta # 1.5
            print(f'Running smoothing on person {person_id}, min_cutoff: {min_cutoff}, beta: {beta}')
            pred_verts, pred_pose, pred_joints3d = smooth_pose(pred_pose, pred_betas,
                                                               min_cutoff=min_cutoff, beta=beta)

        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        joints2d_img_coord = convert_crop_coords_to_orig_img(
            bbox=bboxes,
            keypoints=smpl_joints2d,
            crop_size=224,
        )

        output_dict = {
            #'pred_cam': pred_cam,
            #'orig_cam': orig_cam,
            #'verts': pred_verts,
            #'pose': pred_pose,
            #'betas': pred_betas,
            'joints3d': pred_joints3d,
            #'joints2d': joints2d,
            'joints2d_img_coord': joints2d_img_coord,
            #'bboxes': bboxes,
            'frame_ids': frames,
        }
 
        vibe_results[person_id] = output_dict
    
    #print(vibe_results)
    #first_person_id = list(vibe_results)[0]
    outFile.write(str(num_frames)+"\n")
    #print("num_frames: "+str(num_frames)) 
    #print("kp shape: ")
    #print(vibe_results[1]['joints3d'].shape)
    #print(vibe_results[1]['frame_ids'])
    for i in range(num_frames): # for each frame
        #print("i: "+str(i))
        people_tmp_list = list(vibe_results)
        people_list = people_tmp_list.copy() ##people_tmp_list[:]
        #print(people_list)
        for p in people_tmp_list:
            #print("p: "+str(p))
            if i not in vibe_results[p]['frame_ids']:
                people_list.remove(p)
        #print(people_list)
        num_people = len(people_list)
        #print(num_people)
        outFile.write(str(num_people)+"\n")
        if num_people == 0:
            #print("continue...")
            continue        

        for p in people_list:
           #print("p: "+str(p))
           outFile.write("0 0 0 0 0 0 0 0 0 0"+"\n")
           outFile.write("25\n")
           frame_idx = np.where(vibe_results[p]['frame_ids'] == i)[0][0]
           for x,y,z in vibe_results[p]['joints3d'][frame_idx,0:25:]:
               outFile.write(str(x)+" "+str(y)+" "+str(z)+" 0 0 0 0 0 0 0 0 0\n")   
        '''
        if len(vibe_results) == 1:
            num_people = 1
            if i not in vibe_results[first_person_id]['frame_ids']:
                num_people = 0
        else: ########################## check frame_id
            last_key = list(vibe_results)[-1]
            print("last_key"+str(last_key))
            nose = vibe_results[last_key]['joints2d_img_coord'][0][0][1] #nose position in y coordinate
            toe = vibe_results[last_key]['joints2d_img_coord'][0][23][1] #toe position
            dis = abs(toe-nose) 
            #print("dis: "+str(dis))
            if dis < 0.2*780 or i not in vibe_results[last_key]['frame_ids']:
                num_people = 1
            else:
                num_people = 2
            if (i not in vibe_results[first_person_id]['frame_ids']) and (i not in vibe_results[last_key]['frame_ids']):
                num_people = 0
        if num_people == 0:
            outFile.write(str(num_people)+"\n")
            continue        

        for person_id in list(vibe_results): ##############num_people needs to be modified
            if num_people == 2 and person_id == list(vibe_results)[-1]:
                # do nothing
                zxcv = 0
            else:
                outFile.write(str(num_people)+"\n")
                if num_people == 0:
                    break
            outFile.write("0 0 0 0 0 0 0 0 0 0"+"\n")
            outFile.write("25\n")
            #print("person_id")
            #print(vibe_results[person_id+1]['joints3d'][i][0:25].shape)
            print("i"+str(i))
            print(vibe_results[person_id]['frame_ids'])
            #print( vibe_results[person_id+1]['joints2d_img_coord'].shape)
            if i in vibe_results[person_id]['frame_ids']:
                frame_idx = np.where(vibe_results[person_id]['frame_ids'] == i)[0][0]
                print(frame_idx)
                for x,y,z in vibe_results[person_id]['joints3d'][frame_idx,0:25:]:
                    #print(x, y, z)
                    outFile.write(str(x)+" "+str(y)+" "+str(z)+" 0 0 0 0 0 0 0 0 0\n")
            else:
                print(' ')
                #print(pred_joints3d[i])
        '''
    outFile.close()

    del model

    end = time.time()
    fps = num_frames / (end - vibe_time)

    print(f'VIBE FPS: {fps:.2f}')
    total_time = time.time() - total_time
    print(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
    print(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

    #print(f'Saving output results to \"{os.path.join(output_path, "vibe_output.pkl")}\".')

    #joblib.dump(vibe_results, os.path.join(output_path, "vibe_output.pkl"))
    print(image_folder)
    shutil.rmtree(image_folder)
    print('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_file', type=str,
                        help='input video path or youtube link')

    parser.add_argument('--output_folder', type=str,
                        help='output folder to write results')

    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='batch size of object detector used for bbox tracking')

    parser.add_argument('--staf_dir', type=str, default='/home/mkocabas/developments/openposetrack',
                        help='path to directory STAF pose tracking method installed.')

    parser.add_argument('--vibe_batch_size', type=int, default=450,
                        help='batch size of VIBE')

    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')

    parser.add_argument('--run_smplify', action='store_true',
                        help='run smplify for refining the results, you need pose tracking to enable it')

    parser.add_argument('--no_render', action='store_true',
                        help='disable final rendering of output video.')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')

    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    parser.add_argument('--smooth', action='store_true',
                        help='smooth the results to prevent jitter')

    parser.add_argument('--smooth_min_cutoff', type=float, default=0.004,
                        help='one euro filter min cutoff. '
                             'Decreasing the minimum cutoff frequency decreases slow speed jitter')

    parser.add_argument('--smooth_beta', type=float, default=0.7,
                        help='one euro filter beta. '
                             'Increasing the speed coefficient(beta) decreases speed lag.')

    args = parser.parse_args()

    main(args)
