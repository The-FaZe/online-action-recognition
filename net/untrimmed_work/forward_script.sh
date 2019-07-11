#!/bin/bash

weights = ~/Documents/graduation_project/codes/server_scripts/faze_training/RGB/bninception_results/slurm-223040/ucf101_bninception_128__rgb_checkpoint.pth.tar 

video = ~/Documents/graduation_project/reda.MP4

cd ~/Documents/graduation_project/codes/real-time-action-recognition





python3 -u video_test.py ucf101 RGB  ~/Documents/graduation_project/codes/server_scripts/faze_training/RGB/bninception_results/slurm-223040/ucf101_bninception_128__rgb_checkpoint.pth.tar \
--arch BNInception --classInd_file UCF_lists/classInd.txt --video ~/Documents/graduation_project/reda.MP4 -j 1 --gpus 0
