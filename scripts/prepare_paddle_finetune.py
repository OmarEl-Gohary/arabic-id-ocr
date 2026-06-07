"""
Merge real OCR crops + synthetic data → PaddleOCR recognition fine-tuning format.

Inputs (run these first):
    python scripts/collect_ocr_crops.py       → data/ocr_crops/
    python scripts/generate_arabic_synthetic.py → data/arabic_synthetic/

Output:
    data/paddle_finetune/
        train/images/   ← training images
        val/images/     ← validation images
        train_label.txt ← path<TAB>text  (PaddleOCR format)
        val_label.txt
        arabic_dict.txt ← one character per line (character vocabulary)

Usage:
    python scripts/prepare_paddle_finetune.py
    python scripts/prepare_paddle_finetune.py --aug-factor 7 --val-ratio 0.1
"""

import argparse
import csv
import io
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Augmentation (OpenCV + PIL, tuned for real ID card crops) ────────────────

def augment_crop(img: Image.Image) -> Image.Image:
    """Heavy augmentation for real field crops — simulates bad camera conditions."""

    # Gaussian blur
    if random.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 2.5)))

    # Motion blur
    if random.random() < 0.3:
        k = random.choice([3, 5, 7])
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        arr = np.array(img)
        arr = cv2.filter2D(arr, -1, kernel)
        img = Image.fromarray(arr)

    # Salt-and-pepper
    if random.random() < 0.3:
        arr     = np.array(img)
        density = random.uniform(0.01, 0.05)
        n_px    = int(arr.shape[0] * arr.shape[1] * density)
        for val in (255, 0):
            ys = np.random.randint(0, arr.shape[0], n_px)
            xs = np.random.randint(0, arr.shape[1], n_px)
            arr[ys, xs] = val
        img = Image.fromarray(arr)

    # Perspective skew
    if random.random() < 0.3:
        w, h   = img.size
        skew   = random.uniform(-0.10, 0.10)
        coeffs = (1, skew, -skew * h / 2, 0, 1, 0, 0, 0)
        img = img.transform(
            (w, h), Image.PERSPECTIVE, coeffs,
            Image.BICUBIC, fillcolor=(255, 255, 255),
        )

    # Brightness / contrast
    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))
    if random.random() < 0.4:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.5))

    # Gaussian noise
    if random.random() < 0.3:
        arr   = np.array(img).astype(np.float32)
        arr   = np.clip(arr + np.random.normal(0, random.uniform(5, 25), arr.shape), 0, 255).astype(np.uint8)
        img   = Image.fromarray(arr)

    # Rotation
    if random.random() < 0.35:
        img = img.rotate(random.uniform(-6, 6), fillcolor=(255, 255, 255), expand=False)

    # JPEG compression
    if random.random() < 0.3:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(35, 75))
        buf.seek(0)
        img = Image.open(buf).copy()

    return img


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_char_dict(texts: list[str]) -> list[str]:
    chars = set()
    for t in texts:
        chars.update(t)
    # Sort: ASCII printable first, then Arabic block, then rest
    ascii_chars   = sorted(c for c in chars if ord(c) < 128)
    arabic_chars  = sorted(c for c in chars if 0x0600 <= ord(c) <= 0x06FF)
    other_chars   = sorted(c for c in chars if ord(c) >= 128 and not (0x0600 <= ord(c) <= 0x06FF))
    return ascii_chars + arabic_chars + other_chars


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real-crops",  default="data/ocr_crops",
                   help="Directory produced by collect_ocr_crops.py")
    p.add_argument("--synthetic",   default="data/arabic_synthetic",
                   help="Directory produced by generate_arabic_synthetic.py")
    p.add_argument("--out",         default="data/paddle_finetune",
                   help="Output directory for PaddleOCR training data")
    p.add_argument("--aug-factor",  type=int, default=5,
                   help="How many augmented copies to make per real crop")
    p.add_argument("--val-ratio",   type=float, default=0.1,
                   help="Fraction of data held out for validation")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir   = Path(args.out)
    train_dir = out_dir / "train" / "images"
    val_dir   = out_dir / "val"   / "images"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    records = []   # (src_image_path, text, is_real)

    # ── Load real crops ───────────────────────────────────────────────────────
    real_dir  = Path(args.real_crops)
    real_csv  = real_dir / "labels.csv"
    n_real    = 0
    if real_csv.exists():
        for row in load_csv(real_csv):
            img_path = real_dir / "images" / row["filename"]
            if img_path.exists() and row.get("text", "").strip():
                records.append((img_path, row["text"].strip(), True))
                n_real += 1
        print(f"Real crops loaded   : {n_real}")
    else:
        print(f"WARNING: {real_csv} not found — skipping real crops.")
        print("  Run:  python scripts/collect_ocr_crops.py  first.")

    # ── Load synthetic data ───────────────────────────────────────────────────
    syn_dir  = Path(args.synthetic)
    syn_csv  = syn_dir / "labels.csv"
    n_syn    = 0
    if syn_csv.exists():
        for row in load_csv(syn_csv):
            img_path = syn_dir / "images" / row["filename"]
            if img_path.exists() and row.get("text", "").strip():
                records.append((img_path, row["text"].strip(), False))
                n_syn += 1
        print(f"Synthetic loaded    : {n_syn}")
    else:
        print(f"WARNING: {syn_csv} not found — skipping synthetic data.")
        print("  Run:  python scripts/generate_arabic_synthetic.py  first.")

    if not records:
        print("No data found. Exiting.")
        sys.exit(1)

    # ── Augment real crops and expand dataset ─────────────────────────────────
    expanded = []   # (dest_filename, text)
    counter  = 0

    for src_path, text, is_real in records:
        try:
            base_img = Image.open(src_path).convert("RGB")
        except Exception:
            continue

        copies = args.aug_factor if is_real else 1

        for k in range(copies):
            img = augment_crop(base_img) if (is_real or random.random() < 0.5) else base_img.copy()
            fname = f"rec_{counter:07d}.png"
            expanded.append((img, fname, text))
            counter += 1

    print(f"Total samples       : {counter}  "
          f"(real×{args.aug_factor} + synthetic)")

    # ── Train / val split ─────────────────────────────────────────────────────
    random.shuffle(expanded)
    n_val   = max(1, int(len(expanded) * args.val_ratio))
    val_set = expanded[:n_val]
    trn_set = expanded[n_val:]

    def write_split(samples, img_dir: Path, label_path: Path):
        lines = []
        for img, fname, text in samples:
            dest = img_dir / fname
            img.save(str(dest))
            lines.append(f"images/{fname}\t{text}")
        label_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Writing train ({len(trn_set)}) ...")
    write_split(trn_set, train_dir, out_dir / "train_label.txt")

    print(f"Writing val   ({len(val_set)}) ...")
    write_split(val_set, val_dir, out_dir / "val_label.txt")

    # ── Character dictionary ──────────────────────────────────────────────────
    all_texts  = [text for _, _, text in expanded]
    char_list  = build_char_dict(all_texts)
    dict_path  = out_dir / "arabic_dict.txt"
    dict_path.write_text("\n".join(char_list), encoding="utf-8")

    print(f"\nDone.")
    print(f"  Train label : {out_dir / 'train_label.txt'}  ({len(trn_set)} samples)")
    print(f"  Val label   : {out_dir / 'val_label.txt'}   ({len(val_set)} samples)")
    print(f"  Char dict   : {dict_path}  ({len(char_list)} characters)")
    print(f"\nNext step:")
    print(f"  python -m paddle.distributed.launch --gpus='0' tools/train.py \\")
    print(f"      -c configs/paddle_rec_finetune.yml")


if __name__ == "__main__":
    main()
