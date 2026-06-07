"""
Register the Arabic ID OCR models in the MLflow Model Registry.

Registers two models:
  1. arabic-id-detector  — YOLOv11n field detection model (best.pt)
  2. arabic-ocr-engine   — PaddleOCR Arabic recognition model

Usage:
    python scripts/register_models.py
    python scripts/register_models.py --tracking-uri sqlite:///mlruns.db
    python scripts/register_models.py --stage Production   # promote to Production
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow.tracking import MlflowClient

# ── Model metadata from training results (CLAUDE.md) ──────────────────────────

YOLO_METRICS = {
    "mAP50_overall":    0.937,
    "mAP50_Add1":       0.995,
    "mAP50_Add2":       0.995,
    "mAP50_Front":      0.995,
    "mAP50_Last_Name":  0.995,
    "mAP50_HusbandName":0.995,
    "mAP50_Serial_Num": 0.992,
    "mAP50_First_Name": 0.993,
    "mAP50_ID":         0.975,
    "mAP50_Back":       0.970,
    "mAP50_Gender":     0.892,
    "mAP50_ExpDate":    0.869,
    "mAP50_Religion":   0.867,
    "mAP50_IssueDate":  0.852,
    "mAP50_Job":        0.850,
    "mAP50_Status":     0.814,
    "epochs":           100,
    "img_size":         640,
}

YOLO_TAGS = {
    "architecture":  "YOLOv11n",
    "task":          "object-detection",
    "dataset":       "Egyptian National ID cards",
    "num_classes":   "15",
    "device":        "cpu",
    "framework":     "ultralytics",
    "branch":        "azure-ml",
}

OCR_TAGS = {
    "architecture":  "SVTR_LCNet (PP-OCRv5)",
    "model_name":    "arabic_PP-OCRv5_mobile_rec",
    "task":          "text-recognition",
    "language":      "Arabic",
    "framework":     "paddleocr",
    "input_format":  "BGR numpy array",
}


def _get_or_create_registered_model(client: MlflowClient, name: str, description: str):
    """Create registered model if it doesn't exist yet."""
    try:
        client.create_registered_model(name=name, description=description)
    except Exception:
        pass  # already exists


def register_yolo(client: MlflowClient, weights_path: Path, stage: str) -> str:
    """Log YOLO weights + metrics, register as arabic-id-detector."""
    print("\n--- Registering arabic-id-detector (YOLO) ---")

    # Log the run with weights + metrics
    with mlflow.start_run(run_name="yolo-registration") as run:
        mlflow.set_tags(YOLO_TAGS)
        mlflow.log_metrics(YOLO_METRICS)
        mlflow.log_artifact(str(weights_path), artifact_path="weights")
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri
        print(f"  Run ID       : {run_id}")
        print(f"  Artifact URI : {artifact_uri}")

    # Register the model using client API directly (MLflow 3.x compatible)
    _get_or_create_registered_model(
        client, "arabic-id-detector",
        description=(
            "YOLOv11n trained on Egyptian National ID cards. "
            "Detects 15 field regions. mAP50=0.937 overall. "
            "Trained 100 epochs on Google Colab T4."
        ),
    )

    source = f"{artifact_uri}/weights"
    mv = client.create_model_version(
        name="arabic-id-detector",
        source=source,
        run_id=run_id,
        description="Initial release — azure-ml branch. mAP50=0.937.",
        tags={"ready": "true"},
    )
    print(f"  Version: {mv.version}")

    if stage in ("Staging", "Production"):
        client.transition_model_version_stage(
            name="arabic-id-detector", version=mv.version, stage=stage
        )
        print(f"  Stage  : {stage}")

    return mv.version


def register_ocr(client: MlflowClient, stage: str) -> str:
    """Register PaddleOCR engine metadata as arabic-ocr-engine."""
    print("\n--- Registering arabic-ocr-engine (PaddleOCR) ---")

    with mlflow.start_run(run_name="ocr-registration") as run:
        mlflow.set_tags(OCR_TAGS)
        mlflow.log_params({
            "model_name":   "arabic_PP-OCRv5_mobile_rec",
            "use_gpu":      "false",
            "max_text_len": "50",
        })
        mlflow.log_text(
            "PaddleOCR arabic_PP-OCRv5_mobile_rec — downloaded at runtime by PaddleOCR.\n"
            "Cached at: ~/.paddlex/official_models/arabic_PP-OCRv5_mobile_rec\n",
            "model_info.txt",
        )
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri
        print(f"  Run ID       : {run_id}")

    _get_or_create_registered_model(
        client, "arabic-ocr-engine",
        description=(
            "PaddleOCR Arabic recognition (arabic_PP-OCRv5_mobile_rec). "
            "TextRecognition-only mode — YOLO localises fields first. "
            "Handles Arabic + Arabic-Indic numerals."
        ),
    )

    source = f"{artifact_uri}"
    mv = client.create_model_version(
        name="arabic-ocr-engine",
        source=source,
        run_id=run_id,
        description="PP-OCRv5 mobile Arabic recognition. No fine-tuning yet.",
        tags={"ready": "true"},
    )
    print(f"  Version: {mv.version}")

    if stage in ("Staging", "Production"):
        client.transition_model_version_stage(
            name="arabic-ocr-engine", version=mv.version, stage=stage
        )
        print(f"  Stage  : {stage}")

    return mv.version


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-uri", default="sqlite:///mlruns.db",
                   help="MLflow tracking URI (must be SQLite or DB for registry)")
    p.add_argument("--weights",      default="runs/train/arabic_id_detector/weights/best.pt",
                   help="Path to YOLO weights file")
    p.add_argument("--stage",        default="Staging",
                   choices=["None", "Staging", "Production"],
                   help="Stage to promote model versions to after registration")
    return p.parse_args()


def main():
    args = parse_args()

    weights_path = PROJECT_ROOT / args.weights
    if not weights_path.exists():
        print(f"ERROR: YOLO weights not found at {weights_path}")
        sys.exit(1)

    print(f"MLflow tracking URI : {args.tracking_uri}")
    print(f"YOLO weights        : {weights_path}")
    print(f"Target stage        : {args.stage}")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("model-registry")
    client = MlflowClient()

    yolo_ver = register_yolo(client, weights_path, args.stage)
    ocr_ver  = register_ocr(client, args.stage)

    print("\n--- Summary ---")
    print(f"  arabic-id-detector  v{yolo_ver}  -> {args.stage}")
    print(f"  arabic-ocr-engine   v{ocr_ver}   -> {args.stage}")
    print("\nView in MLflow UI:")
    print("  mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5001")
    print("  → http://localhost:5001  →  Models tab")


if __name__ == "__main__":
    main()
