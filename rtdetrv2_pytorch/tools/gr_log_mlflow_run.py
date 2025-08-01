""" 
Log an experiment folder in output/expname to mlflow.

Clean up the folder before logging (don't upload all checkpoints, just the best, last 
and last 3 checkpoints). I haven't set up the training code to log files cleanly.

Also copy in the training stdout from the piped output into the folder. It contains
the config at the top and readable COCO metrics.

TODO: Clean up the logging process if doing lots of experimenting
"""
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import mlflow

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, default=None, help="Output experiment folder to log to mlflow run")
    parser.add_argument('--artifact-path', type=str, default=None, help="optional sub-folder location in mlflow run")
    parser.add_argument('--run-id', type=str, default=None, help="Log to specific mlflow run")
    parser.add_argument('--run-name', type=str, default=None, help="Name of new run")
    parser.add_argument('--experiment', type=str, default=None, help="Log to mlflow experiment")
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://gr-nuc-visionai:4242")
    
    if args.run_id is None:
        if args.experiment is None:
            raise ValueError("Please specify run-id or give an experiment name")
        mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name, run_id=args.run_id):
        print('Uploading artifacts...')
        mlflow.log_artifacts(args.folder, artifact_path=args.artifact_path)