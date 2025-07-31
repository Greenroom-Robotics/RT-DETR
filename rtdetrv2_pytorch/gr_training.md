# Training GR Model

This is specific instructions on how to training our Greenroom models.

See [rtdetrv2_pytorch/README.md](README.md#usage) for detailed usage documentation 

## Setup

1. Build `rtdetrv2_pytorch` environment
   ```bash
   docker compose build
   ```
2. Run container:
   ```
   docker compose up -d
   docker exec -it rt_detr bash -l
   ```

## Train
The following commands should all be run inside the rt_detr container.
1. Download checkpoint for transfer learning:
    ```bash
    mkdir -p checkpoints
    cd checkpoints
    wget https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
    ```
    Other checkpoints can be found in the [README.md](README.md)

2. Train RT-DETR Model:
    ```
    python3 tools/train.py -c /home/ros/RT-DETR/configs/gr/rtdetrv2_r18vd_120e_rgb_12cls.yml -t /home/ros/RT-DETR/checkpoints/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --use-amp --seed=0
    ```

4. Upload experiment artifacts to MLFlow:
   
   After training, convert the model to onnx and log to MLFlow:
   ```bash
   python3 tools/log_mlflow_model.py --onnx $ONNXFILE --run-id $RUNID
   ```

