"""
Full pipeline test: image → field detection → OCR text extraction → JSON output.

Usage:
    # Test on an image file
    python src/serving/test_inference.py --image path/to/id_card.jpg

    # Capture from camera (press SPACE to capture, Q to quit)
    python src/serving/test_inference.py --camera
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mlflow
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH     = "runs/train/arabic_id_detector/weights/best.pt"
MLFLOW_URI     = "./mlruns"
EXPERIMENT     = "arabic-ocr-inference-tests"
CONF_THRESHOLD = 0.25


def build_pipeline():
    """Load detector + EasyOCR and return pipeline."""
    from src.models.detector import FieldDetector
    from src.models.ocr_engine import create_ocr_engine
    from src.models.pipeline import ArabicIDOCRPipeline

    print("Loading YOLOv11 detector...")
    detector = FieldDetector(
        model_path=MODEL_PATH,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=0.45,
        device="cpu",
    )

    print("Loading EasyOCR (Arabic + English)...")
    ocr = create_ocr_engine(engine="easyocr", languages=["ar", "en"], gpu=False)

    return ArabicIDOCRPipeline(detector=detector, ocr_engine=ocr, min_ocr_confidence=0.3)


def print_result(result: dict):
    """Pretty-print the extraction result."""
    print("\n" + "="*55)
    print(f"  ID Side : {result['id_side'].upper()}")
    print(f"  Fields  : {result['metadata']['num_fields_detected']} detected")
    print(f"  Time    : {result['metadata']['processing_time_s']}s")
    print("="*55)
    print("  Extracted Text Fields:")
    print("-"*55)
    for field, text in sorted(result["fields"].items()):
        val = text if text else "— (not read)"
        print(f"  {field:<15} {val}")
    print("="*55)


def log_to_mlflow(result: dict, image_path: str = None, run_name: str = "inference"):
    """Log the inference result to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=run_name) as run:
        # Params
        mlflow.log_params({
            "model_path":      MODEL_PATH,
            "conf_threshold":  CONF_THRESHOLD,
            "id_side":         result["id_side"],
            "source":          image_path or "camera",
        })

        # Metrics
        fields     = result["fields"]
        n_detected = result["metadata"]["num_fields_detected"]
        n_read     = sum(1 for v in fields.values() if v is not None)
        read_rate  = n_read / n_detected if n_detected else 0

        mlflow.log_metrics({
            "fields_detected":     n_detected,
            "fields_read":         n_read,
            "text_read_rate":      round(read_rate, 3),
            "processing_time_s":   result["metadata"]["processing_time_s"],
        })

        # Per-field read success (1 = text extracted, 0 = not read)
        mlflow.log_metrics({
            f"read_{k}": int(v is not None) for k, v in fields.items()
        })

        # Log full JSON result as artifact
        result_path = f"/tmp/ocr_result_{run.info.run_id[:8]}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(result_path, artifact_path="results")

        # Log annotated image if available
        if image_path and Path(image_path).exists():
            mlflow.log_artifact(image_path, artifact_path="input_images")

        print(f"\n✅ Logged to MLflow  run: {run.info.run_id[:8]}...")
        print(f"   fields_detected={n_detected}  fields_read={n_read}  read_rate={read_rate:.0%}")
        print(f"   View: mlflow ui --backend-store-uri {MLFLOW_URI} --port 5000")

    return run.info.run_id


def test_image(image_path: str, pipeline):
    """Run full pipeline on a saved image."""
    if not Path(image_path).exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    print(f"\nProcessing: {image_path}")
    result = pipeline.process_file(image_path)

    print_result(result)
    log_to_mlflow(result, image_path=image_path, run_name="image-test")

    # Show image with detections using OpenCV
    img = cv2.imread(image_path)
    for det in result.get("raw_detections", []):
        x1, y1, x2, y2 = map(int, det["box_xyxy"])
        text = det.get("ocr_text") or ""
        conf = det.get("ocr_confidence", 0)
        label = f'{det["class_name"]}: {text[:20]}' if text else det["class_name"]

        color = (0, 200, 0) if text else (0, 100, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow("OCR Result — press any key to close", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_camera(pipeline):
    """Capture from webcam, run pipeline on SPACE, log to MLflow."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        sys.exit(1)

    print("\nCamera open. Hold the ID card up to the camera.")
    print("  SPACE → capture & run OCR")
    print("  Q     → quit\n")

    snap_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, "SPACE=capture  Q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Camera — Arabic ID OCR", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            snap_count += 1
            snap_path = f"/tmp/id_capture_{snap_count}.jpg"
            cv2.imwrite(snap_path, frame)
            print(f"\nCaptured frame → running OCR pipeline...")

            cv2.destroyAllWindows()
            cap.release()

            result = pipeline.process_file(snap_path)
            print_result(result)
            log_to_mlflow(result, image_path=snap_path,
                          run_name=f"camera-capture-{snap_count}")

            # Show annotated result
            img = cv2.imread(snap_path)
            for det in result.get("raw_detections", []):
                x1, y1, x2, y2 = map(int, det["box_xyxy"])
                text = det.get("ocr_text") or ""
                label = f'{det["class_name"]}: {text[:18]}' if text else det["class_name"]
                color = (0, 200, 0) if text else (0, 100, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imshow("OCR Result — any key to continue", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Re-open camera for next capture
            cap = cv2.VideoCapture(0)
            print("\nCamera ready. SPACE=capture  Q=quit")

        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended. {snap_count} capture(s) logged to MLflow.")


def main():
    parser = argparse.ArgumentParser(description="Test Arabic ID OCR pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to ID card image")
    group.add_argument("--camera", action="store_true", help="Use webcam")
    args = parser.parse_args()

    pipeline = build_pipeline()

    if args.image:
        test_image(args.image, pipeline)
    else:
        test_camera(pipeline)


if __name__ == "__main__":
    main()
