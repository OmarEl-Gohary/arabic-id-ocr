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


def enhance_for_ocr(image: np.ndarray, min_height: int = 64) -> np.ndarray:
    """
    Preprocessing pipeline for OCR crops:
      1. Upscale if crop is too small (Arabic text needs ≥64px height)
      2. Deskew to correct tilt
      3. CLAHE adaptive contrast
      4. Mild denoising
    """
    # 1. Upscale small crops — tiny text kills OCR accuracy
    h, w = image.shape[:2]
    if h < min_height:
        scale = min_height / h
        image = cv2.resize(image, (int(w * scale), min_height),
                           interpolation=cv2.INTER_CUBIC)

    # 2. Deskew
    image = deskew(image)

    # 3. CLAHE contrast on grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4. Mild denoising
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)


def deskew(image: np.ndarray) -> np.ndarray:
    """Correct skew in a cropped text field image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return image

    h, w = image.shape[:2]
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
