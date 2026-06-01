"""
Crop field regions from ID card images using YOLO, then auto-label
them with PaddleOCR. Saves (crop image + text label) pairs that
will be used to fine-tune a lightweight TrOCR model.

Usage:
    python scripts/collect_ocr_crops.py \
        --images  Egyptain-Person-ID-1/valid/images \
        --out     data/ocr_crops \
        --min-conf 0.70

Output structure:
    data/ocr_crops/
        images/          ← cropped field PNGs
        labels.csv       ← filename, field_name, text, confidence
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import crop_field, enhance_for_ocr, load_image
from src.models.detector import FieldDetector
from src.models.ocr_engine import PaddleOCREngine
from src.models.pipeline import NUMERIC_FIELDS, STRUCTURAL_FIELDS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  default="Egyptain-Person-ID-1",
                   help="Root dataset directory — all train/valid/test splits are used")
    p.add_argument("--out",      default="data/ocr_crops",
                   help="Output directory for crops + labels")
    p.add_argument("--detector", default="runs/train/arabic_id_detector/weights/best.pt")
    p.add_argument("--min-conf", type=float, default=0.65,
                   help="Minimum PaddleOCR confidence to keep a label")
    p.add_argument("--limit",    type=int,   default=0,
                   help="Max images to process per split (0 = all)")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.csv"

    print("Loading YOLO detector...")
    detector = FieldDetector(args.detector, conf_threshold=0.25,
                             iou_threshold=0.45, device="cpu")

    print("Loading PaddleOCR (Arabic)...")
    ocr = PaddleOCREngine(lang="ar")

    # Collect images from all splits (train / valid / test)
    dataset_root = Path(args.dataset)
    image_paths = []
    for split in ("train", "valid", "test"):
        split_dir = dataset_root / split / "images"
        if split_dir.exists():
            imgs = sorted(split_dir.glob("*.jpg"))
            if args.limit:
                imgs = imgs[: args.limit]
            image_paths.extend(imgs)
            print(f"  {split:<6}: {len(imgs)} images")
        else:
            print(f"  {split:<6}: not found, skipping")
    print(f"\nTotal: {len(image_paths)} images\n")

    rows = []
    saved = 0
    skipped_conf = 0
    skipped_empty = 0

    for idx, img_path in enumerate(image_paths, 1):
        try:
            image = load_image(str(img_path))
        except Exception:
            continue

        try:
            detections = detector.detect(image)
        except Exception:
            continue

        for det in detections:
            field_name = det["class_name"]
            if field_name in STRUCTURAL_FIELDS:
                continue

            crop = crop_field(image, det["box_xyxy"])
            if crop.size == 0:
                continue

            field_type = "numeric" if field_name in NUMERIC_FIELDS else "text"
            enhanced   = enhance_for_ocr(crop, field_type=field_type)

            try:
                text, conf = ocr.read_text_with_confidence(enhanced)
            except Exception:
                continue

            text = text.strip()
            if not text:
                skipped_empty += 1
                continue
            if conf < args.min_conf:
                skipped_conf += 1
                continue

            # Save crop as PNG
            stem     = f"{img_path.stem}__{field_name}__{saved:05d}"
            out_path = img_dir / f"{stem}.png"
            cv2.imwrite(str(out_path),
                        cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))

            rows.append({
                "filename":   f"{stem}.png",
                "field_name": field_name,
                "text":       text,
                "confidence": round(conf, 4),
                "source":     img_path.name,
            })
            saved += 1

        if idx % 20 == 0 or idx == len(image_paths):
            print(f"  [{idx}/{len(image_paths)}]  saved={saved}  "
                  f"skipped_low_conf={skipped_conf}  skipped_empty={skipped_empty}")

    # Write CSV
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "field_name",
                                                "text", "confidence", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone.")
    print(f"  Crops saved : {saved}")
    print(f"  Labels CSV  : {labels_path}")
    print(f"  Per-field breakdown:")
    from collections import Counter
    counts = Counter(r["field_name"] for r in rows)
    for fn, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {fn:<16} {n}")


if __name__ == "__main__":
    main()
