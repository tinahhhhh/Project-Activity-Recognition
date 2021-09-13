# Project-Activity-Recognition
This repository includes all scripts for the project "Activity Recognition in Low Resolution". I extract skeletons from videos and train skeletons with MS-G3D model. **The commands include batch size can be adjusted accroding to GPU memory size.**

## Setup Environment
1. git clone https://github.com/mkocabas/VIBE.git
2. git clone https://github.com/kenziyuliu/MS-G3D.git
3. Follow the "Getting Started" part in [VIBE's README file] (https://github.com/mkocabas/VIBE).
4. Installed the [Dependencies] (https://github.com/kenziyuliu/MS-G3D#Dependencies) of MS-G3D. 
5. mv skeleton.py VIBE/
6. mv sk*.sh VIBE/

## Skeleton Extraction
Change directory to VIBE and run the shell scripts parallelly.

1. Modify the "DIR" variable in sk*.sh to specify the video folder path. ( /home/wei/Activity-Recognition/data/input_videos/240/ in the mindgarage server ).&nbsp;
e.g. DIR="/home/wei/Activity-Recognition/data/input_videos/240/1-8".
I divided videos in 5 folders.
2. sh sk.sh (and sh sh sk9.py and more)


## Training MS-G3D
Change directory to MS-G3D and start training.

1. Follow the README file in MS-G3D to do [data preparation] (https://github.com/kenziyuliu/MS-G3D#Data%20Preparation) work. (Put skeletons to specified folders and python3 ntu120_gendata.py) (Download the pretrained model) 
2. Run python3 main.py --config ./config/nturgbd120-cross-subject/train_joint.yaml --work-dir work_dir/ --batch-size 16 --forward-batch-size 8 --num-epoch 100 --weights pretrained-models/ntu120-xsub-joint.pt  