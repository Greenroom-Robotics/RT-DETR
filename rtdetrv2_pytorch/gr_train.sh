#!/bin/bash

# Run this bash script  with `nohup bash gr_train.sh &> train.log &`

# 1. Train model from a pre-trained COCO model (transfer learning)
# Before running this bash script, make sure to edit the following
# - configs/gr/rtdetrv2_r18vd_120e_gr_1280.yml -> Make sure to edit the coco train and test paths, and number of classes, (optional) change the output_dir and number of epochs + train_dataloader policy epochs (i.e. which epoch to stop data aug)

## EXAMPLES
# python3 tools/train.py -c /home/ros/RT-DETR/configs/gr/rtdetrv2_r18vd_120e_rgb_12cls.yml -t /home/ros/RT-DETR/checkpoints/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --use-amp --seed=0
# python3 tools/train.py -c /home/ros/RT-DETR/configs/gr/rtdetrv2_r18vd_120e_ir_3ch.yml -t /home/ros/RT-DETR/checkpoints/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --use-amp --seed=0
# python3 tools/train.py -c /home/ros/RT-DETR/configs/gr/rtdetrv2_r18vd_120e_ir_1ch.yml --use-amp --seed=0

## LARGER MODELS
# python3 tools/train.py -c /home/ros/RT-DETR/configs/gr/rtdetrv2_r50vd_6x_gr_12cls.yml -t /home/ros/RT-DETR/checkpoints/rtdetrv2_r50vd_6x_coco_ema.pth --use-amp --seed=0
# python3 tools/train.py -c /home/ros/RT-DETR/configs/gr/rtdetrv2_r50vd_m_7x_gr_12cls.yml -r /home/ros/RT-DETR/output/rtdetrv2_r50vd_m_7x_d25-001_12cls/last.pth --use-amp --seed=0

# 2. Resume a failed training (due to OOM etc.)
# nohup python3 tools/train.py -c /home/ros/RT-DETR/configs/gr/rtdetrv2_r18vd_120e_gr_1280.yml -r /home/ros/RT-DETR/output/some_epoch_XX.pth --use-amp --seed=0 &> log.txt 2>&1 &
