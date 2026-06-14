"""
Register the Arabic ID OCR models in MLflow Model Registry.

Run once after deployment:
    python scripts/register_model.py --tracking-uri http://10.10.60.10:5000
"""

import argparse
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


YOLO_MODEL_NAME   = "arabic-id-detector"
YOLO_WEIGHTS_PATH = "runs/train/arabic_id_detector/weights/best.pt"

EXPERIMENT_NAME   = "arabic-ocr-registration"


def register_yolo(client: MlflowClient, tracking_uri: str) -> str:
    """Log best.pt as an artifact and register it in the Model Registry."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    weights = Path(YOLO_WEIGHTS_PATH)
    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found at {weights}")

    with mlflow.start_run(run_name="yolo-register") as run:
        mlflow.log_param("architecture", "YOLOv11n")
        mlflow.log_param("num_classes",  15)
        mlflow.log_param("input_size",   640)
        mlflow.log_param("map50",        0.937)
        mlflow.log_param("epochs",       100)
        mlflow.log_param("device",       "cpu")

        # Log weights file as artifact
        mlflow.log_artifact(str(weights), artifact_path="weights")

        run_id = run.info.run_id

    # Register the model from the logged artifact
    model_uri = f"runs:/{run_id}/weights"
    mv = mlflow.register_model(model_uri=model_uri, name=YOLO_MODEL_NAME)
    print(f"Registered {YOLO_MODEL_NAME} — version {mv.version}")

    # Transition to Production
    client.transition_model_version_stage(
        name=YOLO_MODEL_NAME,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Promoted {YOLO_MODEL_NAME} v{mv.version} → Production")
    return mv.version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow tracking server URI",
    )
    args = parser.parse_args()

    print(f"Connecting to MLflow at: {args.tracking_uri}")
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(args.tracking_uri)

    version = register_yolo(client, args.tracking_uri)

    print("\n=== Registration complete ===")
    print(f"Model : {YOLO_MODEL_NAME}")
    print(f"Version: {version}")
    print(f"Stage  : Production")
    print(f"View at: {args.tracking_uri}/#/models/{YOLO_MODEL_NAME}")


if __name__ == "__main__":
    main()
