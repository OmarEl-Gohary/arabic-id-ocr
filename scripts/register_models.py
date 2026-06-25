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
from typing import Optional

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

# File extensions that are part of a PaddleOCR inference model directory
_PADDLE_INFERENCE_EXTS = {".pdmodel", ".pdiparams", ".pdiparams.info", ".yml", ".yaml", ".json", ".txt"}


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


def register_ocr(
    client: MlflowClient,
    stage: str,
    model_dir: Optional[Path] = None,
    cer: Optional[float] = None,
) -> str:
    """Register PaddleOCR engine as arabic-ocr-engine.

    If model_dir is given (exported inference directory from Colab fine-tuning),
    the actual model files are logged as artifacts and CER can be recorded.
    Otherwise only metadata is logged (pretrained model downloaded at runtime).
    """
    print("\n--- Registering arabic-ocr-engine (PaddleOCR) ---")

    is_finetuned = model_dir is not None
    variant = "finetuned" if is_finetuned else "pretrained"

    tags = {**OCR_TAGS, "model_variant": variant}

    params = {
        "model_name":    "arabic_PP-OCRv5_mobile_rec",
        "use_gpu":       "false",
        "max_text_len":  "40",
        "model_variant": variant,
    }
    if is_finetuned:
        params["model_dir"] = str(model_dir)

    metrics: dict = {}
    if cer is not None:
        metrics["CER"] = cer
        print(f"  CER          : {cer:.4f}")

    run_name = f"ocr-registration-{variant}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)

        if is_finetuned:
            # Log all inference model files (pdmodel, pdiparams, config, dict …)
            files_logged = 0
            for fpath in sorted(model_dir.rglob("*")):
                if fpath.is_file() and fpath.suffix in _PADDLE_INFERENCE_EXTS:
                    mlflow.log_artifact(str(fpath), artifact_path="paddleocr_model")
                    files_logged += 1
            if files_logged == 0:
                print(f"  WARNING: no inference files found in {model_dir}")
            else:
                print(f"  Logged {files_logged} model file(s) from {model_dir}")
        else:
            mlflow.log_text(
                "PaddleOCR arabic_PP-OCRv5_mobile_rec — downloaded at runtime.\n"
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

    source = f"{artifact_uri}/paddleocr_model" if is_finetuned else artifact_uri
    desc = (
        f"Fine-tuned PP-OCRv5 Arabic — CER={cer:.4f}" if (is_finetuned and cer is not None)
        else "Fine-tuned PP-OCRv5 Arabic recognition" if is_finetuned
        else "PP-OCRv5 mobile Arabic recognition. Pretrained — no fine-tuning."
    )
    mv = client.create_model_version(
        name="arabic-ocr-engine",
        source=source,
        run_id=run_id,
        description=desc,
        tags={"ready": "true", "variant": variant},
    )
    print(f"  Version: {mv.version}")

    if stage in ("Staging", "Production"):
        client.transition_model_version_stage(
            name="arabic-ocr-engine", version=mv.version, stage=stage
        )
        print(f"  Stage  : {stage}")

    return mv.version


def parse_args():
    p = argparse.ArgumentParser(
        description="Register YOLO + PaddleOCR models in MLflow Model Registry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register pretrained models (metadata only for PaddleOCR):
  python scripts/register_models.py

  # Register fine-tuned PaddleOCR after Colab training:
  python scripts/register_models.py \\
      --paddleocr-model-dir models/arabic_id_rec_finetuned \\
      --cer 0.085

  # Register against a remote MLflow server:
  python scripts/register_models.py --tracking-uri http://10.10.60.10:5000
""",
    )
    p.add_argument("--tracking-uri", default="sqlite:///mlruns.db",
                   help="MLflow tracking URI (must be SQLite or DB for registry)")
    p.add_argument("--weights",      default="runs/train/arabic_id_detector/weights/best.pt",
                   help="Path to YOLO weights file")
    p.add_argument("--stage",        default="Staging",
                   choices=["None", "Staging", "Production"],
                   help="Stage to promote model versions to after registration")
    p.add_argument("--paddleocr-model-dir", default=None,
                   help=(
                       "Path to exported PaddleOCR inference directory "
                       "(e.g. models/arabic_id_rec_finetuned). "
                       "If omitted, only metadata is logged (pretrained model)."
                   ))
    p.add_argument("--cer", type=float, default=None,
                   help="Character Error Rate from fine-tuning evaluation (optional, 0–1)")
    p.add_argument("--skip-yolo", action="store_true",
                   help="Skip YOLO registration (useful when re-registering OCR only)")
    p.add_argument("--skip-ocr",  action="store_true",
                   help="Skip PaddleOCR registration (useful when re-registering YOLO only)")
    return p.parse_args()


def main():
    args = parse_args()

    weights_path = PROJECT_ROOT / args.weights
    paddleocr_dir: Optional[Path] = (
        Path(args.paddleocr_model_dir).resolve() if args.paddleocr_model_dir else None
    )

    if not args.skip_yolo and not weights_path.exists():
        print(f"ERROR: YOLO weights not found at {weights_path}")
        sys.exit(1)

    if paddleocr_dir and not paddleocr_dir.exists():
        print(f"ERROR: PaddleOCR model dir not found at {paddleocr_dir}")
        sys.exit(1)

    print(f"MLflow tracking URI : {args.tracking_uri}")
    if not args.skip_yolo:
        print(f"YOLO weights        : {weights_path}")
    if paddleocr_dir:
        print(f"PaddleOCR model dir : {paddleocr_dir}")
        if args.cer is not None:
            print(f"CER                 : {args.cer}")
    print(f"Target stage        : {args.stage}")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("model-registry")
    client = MlflowClient()

    yolo_ver = ocr_ver = None

    if not args.skip_yolo:
        yolo_ver = register_yolo(client, weights_path, args.stage)

    if not args.skip_ocr:
        ocr_ver = register_ocr(client, args.stage, paddleocr_dir, args.cer)

    print("\n--- Summary ---")
    if yolo_ver:
        print(f"  arabic-id-detector  v{yolo_ver}  -> {args.stage}")
    if ocr_ver:
        print(f"  arabic-ocr-engine   v{ocr_ver}   -> {args.stage}")
    print("\nView in MLflow UI:")
    print("  mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5001")
    print("  → http://localhost:5001  →  Models tab")


if __name__ == "__main__":
    main()
