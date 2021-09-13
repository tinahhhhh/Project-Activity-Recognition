#!/bin/bash

DIR="../data/input_videos/420/18-24"
for entry in "$DIR"/*
    do
        echo "$entry"
        python3 skeleton.py --vid_file "$entry" --output_folder output/ --vibe_batch_size 32 --tracker_batch_size 2 --no_render
    done
