"""
Test the OCR API with all images in a folder.

Usage:
    python test_images.py
    python test_images.py --folder C:\path\to\images
    python test_images.py --image C:\path\to\single_image.jpg
"""

import argparse
import base64
import json
import sys
from pathlib import Path

# Force UTF-8 output on Windows so Arabic characters print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests

API_URL          = "http://localhost:8000/v1/ocr/base64"
API_COMBINED_URL = "http://localhost:8000/v1/ocr/combined"
IMG_FOLDER = Path(r"C:\Users\O.MELgohary\Desktop\OCR\images")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def test_image(image_path: Path) -> dict:
    """Send a single image to the OCR API and return the result."""
    response = requests.post(
        API_URL,
        json={"image": _b64(image_path), "include_raw": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def test_combined(front_path: Path, back_path: Path) -> dict:
    """Send front + back images together and return the merged result."""
    response = requests.post(
        API_COMBINED_URL,
        json={"front": _b64(front_path), "back": _b64(back_path), "include_raw": False},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def print_result(image_path: Path, result: dict):
    print(f"\n{'='*60}")
    print(f"Image      : {image_path.name}")
    print(f"Description: {result.get('description', '')}")
    print("-" * 60)
    for prop in result.get("properties", []):
        name  = prop.get("name", "")
        value = prop.get("value")
        if value:
            print(f"  {name:<20}: {value}")
        else:
            print(f"  {name:<20}: null")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test OCR API with images")
    parser.add_argument("--folder",   type=Path, default=IMG_FOLDER,
                        help="Folder containing images to test (single mode)")
    parser.add_argument("--image",    type=Path, default=None,
                        help="Single image to test")
    parser.add_argument("--front",    type=Path, default=None,
                        help="Front image for combined endpoint")
    parser.add_argument("--back",     type=Path, default=None,
                        help="Back image for combined endpoint")
    args = parser.parse_args()

    passed = 0
    failed = 0

    # ── Combined mode ──────────────────────────────────────────────────────────
    if args.front and args.back:
        print(f"Combined mode: front={args.front.name}  back={args.back.name}")
        try:
            result = test_combined(args.front, args.back)
            print_result(Path("combined"), result)
            out_path = Path("combined_response.json")
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [saved] {out_path}")
            passed += 1
        except Exception as e:
            print(f"\n[ERROR] combined: {e}")
            failed += 1
        print(f"\nDone — {passed} passed, {failed} failed")
        return

    # ── Single / folder mode ───────────────────────────────────────────────────
    if args.image:
        images = [args.image]
    else:
        folder = args.folder
        images = sorted([p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS])
        if not images:
            print(f"No images found in {folder}")
            sys.exit(1)
        print(f"Found {len(images)} image(s) in {folder}")

    for img_path in images:
        try:
            result = test_image(img_path)
            print_result(img_path, result)
            out_path = Path(img_path.stem + "_response.json")
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [saved] {out_path}")
            passed += 1
        except Exception as e:
            print(f"\n[ERROR] {img_path.name}: {e}")
            failed += 1

    print(f"\nDone — {passed} passed, {failed} failed")


if __name__ == "__main__":
    main()
