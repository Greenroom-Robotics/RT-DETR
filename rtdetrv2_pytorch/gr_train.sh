#!/bin/bash

# 1. Train model from a pre-trained COCO model (transfer learning)
# Before running this bash script, make sure to edit the following
# - configs/dataset/gr_detection.yml -> Make sure to edit the coco train and test paths, and number of classes 
# - configs/gr/rtdetrv2_r18vd_120e_gr_1280.yml -> (Optional) Change the output_dir and number of epochs + train_dataloader policy epochs (i.e. which epoch to stop data aug)
nohup python3 tools/train.py -c /home/ros/RT-DETR/configs/gr/rtdetrv2_r18vd_120e_gr_1280.yml -t /home/ros/RT-DETR/checkpoints/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --use-amp --seed=0 &> log.txt 2>&1 &


# 2. Resume a failed training (due to OOM etc.)
# nohup python3 tools/train.py -c /home/ros/RT-DETR/configs/gr/rtdetrv2_r18vd_120e_gr_1280.yml -r /home/ros/RT-DETR/output/some_epoch_XX.pth --use-amp --seed=0 &> log.txt 2>&1 &
