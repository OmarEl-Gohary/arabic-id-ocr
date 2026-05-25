"""Training script for YOLOv11 field detector with MLflow tracking."""

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import yaml
from ultralytics import YOLO


def train(config_path: str = "configs/training.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mlflow_cfg = cfg["mlflow"]
    train_cfg = cfg["training"]
    aug_cfg = cfg["augmentation"]
    out_cfg = cfg["output"]

    # Azure ML auto-injects MLFLOW_TRACKING_URI; fall back to config for local runs
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", mlflow_cfg["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(tags=mlflow_cfg.get("run_tags", {})) as run:
        print(f"MLflow run ID: {run.info.run_id}")

        mlflow.log_params({
            "model": cfg.get("model", {}).get("model_name", "yolo11n"),
            "epochs": train_cfg["epochs"],
            "batch_size": train_cfg["batch_size"],
            "img_size": train_cfg["img_size"],
            "lr0": train_cfg["lr0"],
            "dataset": cfg["data"]["dataset_path"],
        })

        model = YOLO("yolo11n.pt")

        results = model.train(
            data=cfg["data"]["dataset_path"],
            epochs=train_cfg["epochs"],
            batch=train_cfg["batch_size"],
            imgsz=train_cfg["img_size"],
            lr0=train_cfg["lr0"],
            lrf=train_cfg["lrf"],
            momentum=train_cfg["momentum"],
            weight_decay=train_cfg["weight_decay"],
            warmup_epochs=train_cfg["warmup_epochs"],
            patience=train_cfg["patience"],
            save_period=train_cfg["save_period"],
            workers=train_cfg["workers"],
            device=train_cfg["device"],
            project=out_cfg["project"],
            name=out_cfg["name"],
            exist_ok=out_cfg["exist_ok"],
            # Augmentation
            hsv_h=aug_cfg["hsv_h"],
            hsv_s=aug_cfg["hsv_s"],
            hsv_v=aug_cfg["hsv_v"],
            degrees=aug_cfg["degrees"],
            translate=aug_cfg["translate"],
            scale=aug_cfg["scale"],
            shear=aug_cfg["shear"],
            flipud=aug_cfg["flipud"],
            fliplr=aug_cfg["fliplr"],
            mosaic=aug_cfg["mosaic"],
        )

        # Log final metrics
        if hasattr(results, "results_dict"):
            metrics = results.results_dict
            mlflow.log_metrics({
                "mAP50": metrics.get("metrics/mAP50(B)", 0),
                "mAP50_95": metrics.get("metrics/mAP50-95(B)", 0),
                "precision": metrics.get("metrics/precision(B)", 0),
                "recall": metrics.get("metrics/recall(B)", 0),
                "box_loss": metrics.get("train/box_loss", 0),
                "cls_loss": metrics.get("train/cls_loss", 0),
            })

        # Log best weights as artifact
        best_weights = Path(out_cfg["project"]) / out_cfg["name"] / "weights" / "best.pt"
        if best_weights.exists():
            mlflow.log_artifact(str(best_weights), artifact_path="weights")
            print(f"Best weights logged: {best_weights}")

        print(f"Training complete. Run ID: {run.info.run_id}")
        return run.info.run_id


def main():
    parser = argparse.ArgumentParser(description="Train Arabic ID OCR detector")
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
