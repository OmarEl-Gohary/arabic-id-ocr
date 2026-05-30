"""
Live camera inference test with MLflow monitoring.

Usage:
    python src/serving/camera_test.py

Keyboard controls:
    S  — save snapshot + log annotated frame to MLflow
    Q  — quit and log session summary
"""

import os
import sys
import time
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import mlflow
import numpy as np
import yaml
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = "runs/train/arabic_id_detector/weights/best.pt"
CONF_THRESHOLD  = 0.25
LOG_EVERY_N     = 15          # log metrics every N frames (~0.5s at 30fps)
MLFLOW_URI      = "./mlruns"
EXPERIMENT      = "arabic-ocr-camera-monitoring"

# 15 field classes
CLASSES = [
    "Add1", "Add2", "Back", "ExpDate", "First_Name", "Front",
    "Gender", "HusbandName", "ID", "IssueDate", "Job",
    "Last_Name", "Religion", "Serial_Num", "Status"
]


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)


def load_model():
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Place best.pt at that path and rerun.")
        sys.exit(1)
    print(f"Loading model: {MODEL_PATH}")
    return YOLO(MODEL_PATH)


def draw_overlay(frame, results, fps, frame_count, session_detections):
    """Draw bounding boxes + HUD overlay on the frame."""
    annotated = results[0].plot(
        conf=True, labels=True, line_width=2, font_size=0.5
    )

    h, w = annotated.shape[:2]

    # Semi-transparent HUD background
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (300, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

    # HUD text
    boxes   = results[0].boxes
    n_dets  = len(boxes) if boxes is not None else 0
    avg_conf = float(boxes.conf.mean()) if n_dets > 0 else 0.0

    lines = [
        f"Frame:      {frame_count}",
        f"FPS:        {fps:.1f}",
        f"Detections: {n_dets}",
        f"Avg conf:   {avg_conf:.2f}",
        f"Total sess: {sum(session_detections.values())}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(annotated, line, (8, 22 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1)

    # Controls legend
    cv2.putText(annotated, "S=snapshot  Q=quit",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return annotated


def run_camera_session():
    setup_mlflow()
    model = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera (index 0).")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "="*50)
    print("  Arabic ID OCR — Live Camera Test")
    print("="*50)
    print("  S  →  save snapshot to MLflow")
    print("  Q  →  quit & log session summary")
    print("="*50 + "\n")

    with mlflow.start_run(run_name="camera-session") as run:
        mlflow.log_params({
            "model_path":       MODEL_PATH,
            "conf_threshold":   CONF_THRESHOLD,
            "input_source":     "webcam-0",
            "log_every_n_frames": LOG_EVERY_N,
        })
        print(f"MLflow run ID: {run.info.run_id}")
        print(f"Track at:      {MLFLOW_URI}\n")

        frame_count       = 0
        snapshot_count    = 0
        session_detections = defaultdict(int)   # class_name → total detections
        inference_times   = []
        snapshot_dir      = tempfile.mkdtemp()

        prev_time = time.time()
        fps       = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed — exiting.")
                break

            # ── Inference ────────────────────────────────────────────────────
            t0      = time.time()
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            inf_ms  = (time.time() - t0) * 1000
            inference_times.append(inf_ms)

            # ── FPS ──────────────────────────────────────────────────────────
            now      = time.time()
            fps      = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # ── Parse detections ─────────────────────────────────────────────
            boxes = results[0].boxes
            frame_class_counts = defaultdict(int)
            frame_confidences  = defaultdict(list)

            if boxes is not None and len(boxes):
                for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                    name = CLASSES[int(cls_id)] if int(cls_id) < len(CLASSES) else str(int(cls_id))
                    frame_class_counts[name] += 1
                    frame_confidences[name].append(conf)
                    session_detections[name] += 1

            # ── Log to MLflow every N frames ─────────────────────────────────
            if frame_count % LOG_EVERY_N == 0:
                step = frame_count // LOG_EVERY_N

                # Core metrics
                mlflow.log_metrics({
                    "inference_time_ms": inf_ms,
                    "fps":               fps,
                    "num_detections":    len(boxes) if boxes is not None else 0,
                    "avg_confidence":    float(boxes.conf.mean()) if boxes is not None and len(boxes) else 0.0,
                }, step=step)

                # Per-class detection counts this frame
                per_class = {f"detected_{cls}": frame_class_counts.get(cls, 0)
                             for cls in CLASSES}
                mlflow.log_metrics(per_class, step=step)

                # Per-class avg confidence this frame
                conf_metrics = {
                    f"conf_{cls}": float(np.mean(frame_confidences[cls]))
                    for cls in frame_confidences
                }
                if conf_metrics:
                    mlflow.log_metrics(conf_metrics, step=step)

            # ── Draw & display ───────────────────────────────────────────────
            display = draw_overlay(frame, results, fps, frame_count, session_detections)
            cv2.imshow("Arabic ID OCR — Camera Test  (S=snapshot Q=quit)", display)

            # ── Key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') or key == ord('S'):
                # Save annotated snapshot as MLflow artifact
                snap_path = str(Path(snapshot_dir) / f"snapshot_{snapshot_count:04d}.jpg")
                cv2.imwrite(snap_path, display)
                mlflow.log_artifact(snap_path, artifact_path="snapshots")
                snapshot_count += 1
                print(f"Snapshot saved → MLflow artifact (snapshot #{snapshot_count})")

            elif key == ord('q') or key == ord('Q'):
                print("\nQuitting — logging session summary...")
                break

            frame_count += 1

        # ── Session summary ───────────────────────────────────────────────────
        cap.release()
        cv2.destroyAllWindows()

        if inference_times:
            mlflow.log_metrics({
                "session_total_frames":   frame_count,
                "session_avg_inf_ms":     float(np.mean(inference_times)),
                "session_avg_fps":        float(1000 / np.mean(inference_times)),
                "session_total_snapshots": snapshot_count,
            })

        # Per-class session totals
        mlflow.log_metrics({f"session_total_{k}": v
                            for k, v in session_detections.items()})

        print("\n" + "="*50)
        print("  Session Summary")
        print("="*50)
        print(f"  Frames processed : {frame_count}")
        print(f"  Avg inference    : {np.mean(inference_times):.1f} ms")
        print(f"  Avg FPS          : {1000/np.mean(inference_times):.1f}")
        print(f"  Snapshots saved  : {snapshot_count}")
        print(f"\n  Detections per class:")
        for cls, count in sorted(session_detections.items(), key=lambda x: -x[1]):
            print(f"    {cls:<15} {count}")
        print("="*50)
        print(f"\nMLflow run complete. View with:")
        print(f"  mlflow ui --backend-store-uri {MLFLOW_URI} --port 5000")


if __name__ == "__main__":
    run_camera_session()
