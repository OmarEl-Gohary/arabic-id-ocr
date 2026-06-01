"""Image preprocessing utilities for Arabic ID OCR pipeline."""

from typing import Optional, Tuple

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_with_padding(
    image: np.ndarray,
    target_size: int = 640,
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Letterbox resize — returns (resized_image, scale, (pad_w, pad_h))."""
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2

    padded = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    return padded, scale, (pad_w, pad_h)


def crop_field(
    image: np.ndarray,
    box_xyxy: Tuple[float, float, float, float],
    padding: int = 4,
) -> np.ndarray:
    """Crop a field from the image using xyxy bounding box with optional padding."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(w, int(x2) + padding)
    y2 = min(h, int(y2) + padding)
    return image[y1:y2, x1:x2]


def enhance_for_ocr(
    image: np.ndarray,
    min_height: int = 96,
    border: int = 10,
    field_type: str = "text",
) -> np.ndarray:
    """
    Preprocessing pipeline for OCR crops.

    Args:
        image:      RGB crop from the ID card.
        min_height: Upscale until height ≥ this value (Arabic needs ≥96 px).
        border:     White pixel border added around the image (Tesseract
                    works better when text is not flush against the edge).
        field_type: "numeric"  → extra contrast / morphological closing to
                                  strengthen digit strokes before binarisation.
                    "text"     → standard pipeline (default).

    Steps:
      1. Upscale (bicubic) if crop height < min_height
      2. Deskew
      3. Unsharp-mask sharpening
      4. CLAHE adaptive contrast
      5. Mild denoising
      6. [numeric only] morphological closing to join broken digit strokes
      7. White border padding
    """
    # 1. Upscale — target at least 2× the minimum so Tesseract has enough detail
    h, w = image.shape[:2]
    target_h = max(min_height, h)          # never shrink
    if h < min_height:
        scale = min_height / h
        image = cv2.resize(
            image, (max(1, int(w * scale)), min_height),
            interpolation=cv2.INTER_CUBIC,
        )
        h, w = image.shape[:2]

    # Additional 2× upscale when the crop is still very small
    if h < 128:
        image = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # 2. Deskew
    image = deskew(image)

    # 3. Unsharp-mask sharpening (accentuates stroke edges)
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2)
    image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    # 4. CLAHE contrast on grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 5. Mild denoising
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    # 6. Numeric-specific: morphological closing joins broken digit segments
    if field_type == "numeric":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    # 7. White border — Tesseract misreads characters at image edges
    if border > 0:
        denoised = cv2.copyMakeBorder(
            denoised, border, border, border, border,
            cv2.BORDER_CONSTANT, value=255,
        )

    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)


def deskew(image: np.ndarray, max_angle: float = 8.0) -> np.ndarray:
    """
    Correct slight skew in a cropped text field image.

    Safety limits applied:
    - Only corrects angles in (0.5°, max_angle°) — larger angles usually mean
      minAreaRect picked up the wrong orientation on Arabic character strokes,
      which would catastrophically tilt the image.
    - Requires ≥ 100 dark pixels for a reliable angle estimate.
    """
    h, w = image.shape[:2]
    # Deskew is unreliable on very small or very narrow crops
    if h < 20 or w < 20:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:           # too few text pixels → skip
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle

    # Reject large rotations — they almost always indicate a wrong detection
    if abs(angle) < 0.5 or abs(angle) > max_angle:
        return image

    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize to float32 in [0, 1]."""
    return image.astype(np.float32) / 255.0


def bytes_to_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image bytes")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
