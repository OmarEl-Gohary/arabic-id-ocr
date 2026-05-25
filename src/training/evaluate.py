"""Evaluation script — runs YOLO validation and logs metrics to MLflow."""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import mlflow
import yaml
from ultralytics import YOLO


def evaluate(
    model_path: str,
    data_yaml: str,
    split: str = "test",
    img_size: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    run_id: Optional[str] = None,
    tracking_uri: str = "./mlruns",
) -> Dict:
    mlflow.set_tracking_uri(tracking_uri)

    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=img_size,
        conf=conf,
        iou=iou,
        verbose=True,
    )

    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class_mAP50": {
            k: float(v) for k, v in zip(metrics.names.values(), metrics.box.maps)
        },
    }

    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({f"eval_{k}": v for k, v in results.items()
                                 if isinstance(v, float)})
            for cls, score in results["per_class_mAP50"].items():
                mlflow.log_metric(f"eval_mAP50_{cls}", score)

    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Arabic ID OCR detector")
    parser.add_argument("--model", required=True, help="Path to .pt weights")
    parser.add_argument("--data", default="Egyptain-Person-ID-1/data.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
