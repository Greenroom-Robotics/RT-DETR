import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import mlflow
import datetime
import numpy as np

def get_signature(input_shape=(1280,1280)):
    from mlflow.types import Schema, TensorSpec
    from mlflow.models import ModelSignature

    resize_h, resize_w = input_shape
    input_schema = Schema([TensorSpec(type=np.dtype(np.float32), shape=(-1, 3, resize_h, resize_w), name="input"), 
                           TensorSpec(type=np.dtype(np.float32), shape=(-1, 2), name="original_size")])
    output_schema = Schema([
        TensorSpec(type=np.dtype(np.float32), shape=(-1, 300, 4), name="labels"),  # First output tensor
        TensorSpec(type=np.dtype(np.float32), shape=(-1, 300, 4), name="boxes"),  # Second output tensor
        TensorSpec(type=np.dtype(np.float32), shape=(-1, 300), name="scores"),     # Third output tensor
    ])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)       
    return signature

    
def log_torch(args):
    import torch
    from src.core import YAMLConfig

    print(f'Loading model: {args.ckpt}')

    # Load pytorch model
    cfg = YAMLConfig(args.config, resume=args.ckpt)

    checkpoint = torch.load(args.ckpt, map_location='cpu') 
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']

    cfg.model.load_state_dict(state)

    class Model(torch.nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()

    # Log to mlflow
    input_shape = (1280,1280)
    metadata = {"date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "classes": args.class_names,
                "model": "RT-DETR",
                "input_shape": input_shape
                }
    
    signature = get_signature(input_shape=input_shape)

    print('Logging to MLflow...')
    with mlflow.start_run(run_id=args.run_id):
        mlflow.pytorch.log_model(model, "torch_model", signature=signature, metadata=metadata)

def log_onnx(args):
    import onnx
    import json

    print(f'Loading model: {args.onnx}')
    model = onnx.load(args.onnx) 
    metadata = {mp.key:mp.value for mp in model.metadata_props}
    metadata['input_shape'] = json.loads(metadata['input_shape'])
    metadata['classes'] = json.loads(metadata['classes'])
    signature = get_signature(input_shape=metadata['input_shape'])

    print('Logging to MLflow...')
    with mlflow.start_run(run_id=args.run_id):
        mlflow.onnx.log_model(model, "onnx_model", signature=signature, metadata=metadata)  
 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--onnx', type=str, default=None)
    parser.add_argument('--class_names', nargs='+', default=['marine_mammal', 'marker', 'unknown', 'vessel'], help='class list in the same order as class enum')
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--experiment', type=str, default=None)
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://gr-nuc-visionai:4242")
    
    if args.run_id is None:
        if args.experiment is None:
            raise ValueError("Please specify run-id or give an experiment name")
        mlflow.set_experiment(args.experiment)

    if args.ckpt:
        log_torch(args)
    if args.onnx:
        log_onnx(args)
