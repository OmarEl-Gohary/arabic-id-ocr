"""End-to-end pipeline: image → field detection → OCR → JSON."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml

from src.data.preprocess import bytes_to_image, crop_field, enhance_for_ocr, load_image, preprocess_full_image
from src.models.detector import FieldDetector
from src.models.ocr_engine import BaseOCREngine, TesseractOCREngine, create_ocr_engine
from src.models.postprocess import postprocess_fields

logger = logging.getLogger(__name__)

STRUCTURAL_FIELDS = {"Front", "Back"}
NUMERIC_FIELDS    = {"ID", "Serial_Num", "ExpDate", "IssueDate"}


class ArabicIDOCRPipeline:
    """
    Two-stage pipeline:
    1. YOLOv11 detects field bounding boxes on the ID card image.
    2. EasyOCR / PaddleOCR reads Arabic+English text from each cropped field.
    Outputs a structured JSON dict with field names as keys.
    """

    def __init__(
        self,
        detector: FieldDetector,
        ocr_engine: BaseOCREngine,
        min_ocr_confidence: float = 0.3,
        enhance_crops: bool = True,
    ):
        self.detector = detector
        self.ocr = ocr_engine
        self.min_ocr_confidence = min_ocr_confidence
        self.enhance_crops = enhance_crops

    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Process a single RGB image and return structured JSON."""
        start = time.time()

        detections = self.detector.detect(image)
        id_side = self._identify_side(detections)

        fields: Dict[str, Any] = {}
        raw_detections = []

        for det in detections:
            field_name = det["class_name"]

            if field_name in STRUCTURAL_FIELDS:
                continue

            crop = crop_field(image, det["box_xyxy"])
            if crop.size == 0:
                continue

            if self.enhance_crops:
                field_type = "numeric" if field_name in NUMERIC_FIELDS else "text"
                crop = enhance_for_ocr(crop, field_type=field_type)

            # Pass field_name to Tesseract so it can select the best PSM/binarisation
            if isinstance(self.ocr, TesseractOCREngine):
                text, conf = self.ocr.read_text_with_confidence(crop, field_name=field_name)
            else:
                text, conf = self.ocr.read_text_with_confidence(crop)

            if conf < self.min_ocr_confidence:
                text = None

            fields[field_name] = text
            raw_detections.append({
                **det,
                "ocr_text": text,
                "ocr_confidence": conf,
            })

            # ── Job crop → split top/bottom for Job_Name / Company_Name ──────
            # Egyptian ID job section has two lines:
            #   top half  → الوظيفة  (occupation)   → Job
            #   bottom half → جهة العمل (employer) → Company
            if field_name == "Job":
                job_text, company_text = self._split_job_company(crop)
                if job_text:
                    fields["Job"] = job_text          # override with top-half read
                if company_text:
                    fields["Company"] = company_text

        # ── Field validation + retry ──────────────────────────────────────────
        # For each critical field, if the first OCR pass is invalid/empty,
        # re-crop the same bounding box and retry with different preprocessing.
        fields = self._retry_field(image, fields, raw_detections, "ID",      self._is_valid_id,   self._normalise_id)
        fields = self._retry_field(image, fields, raw_detections, "Job",     self._is_valid_text, self._normalise_text)
        fields = self._retry_field(image, fields, raw_detections, "HusbandName", self._is_valid_text, self._normalise_text)

        # Apply field-specific post-processing (numeral normalisation,
        # date formatting, gender/religion vocabulary matching, etc.)
        fields = postprocess_fields(fields)

        elapsed = round(time.time() - start, 3)

        return {
            "id_side": id_side,
            "fields": fields,
            "metadata": {
                "num_fields_detected": len(raw_detections),
                "processing_time_s": elapsed,
            },
            "raw_detections": raw_detections,
        }

    def process_file(self, image_path: str) -> Dict[str, Any]:
        image = load_image(image_path)
        result = self.process_image(image)
        result["metadata"]["source"] = image_path
        return result

    def process_bytes(self, data: bytes) -> Dict[str, Any]:
        image = bytes_to_image(data)
        return self.process_image(image)

    # ── Validators ────────────────────────────────────────────────────────────

    @staticmethod
    def _is_valid_id(value: Optional[str]) -> bool:
        """Exactly 14 ASCII digits."""
        return bool(value and len(value) == 14 and value.isdigit())

    @staticmethod
    def _is_valid_text(value: Optional[str]) -> bool:
        """At least 2 Arabic characters (rejects null, garbage, single chars)."""
        if not value:
            return False
        arabic_chars = sum(1 for c in value if "؀" <= c <= "ۿ")
        return arabic_chars >= 2

    # ── Normalisers (quick pre-postprocess clean used during retry) ───────────

    @staticmethod
    def _normalise_id(text: Optional[str]) -> Optional[str]:
        import re
        from src.models.postprocess import normalize_numerals, strip_bidi
        if not text:
            return None
        digits = re.sub(r"\D", "", normalize_numerals(strip_bidi(text)))
        return digits if digits else None

    @staticmethod
    def _normalise_text(text: Optional[str]) -> Optional[str]:
        from src.models.postprocess import strip_bidi
        return strip_bidi(text).strip() if text else None

    # ── Generic field retry ───────────────────────────────────────────────────

    def _retry_field(
        self,
        image: np.ndarray,
        fields: Dict[str, Any],
        raw_detections: List[Dict],
        field_name: str,
        validator,
        normaliser,
    ) -> Dict[str, Any]:
        """
        Generic retry for any field whose first-pass value fails validation.

        Strategies tried in order:
          1. Raw crop × 3 upscale  — removes CLAHE/sharpen artefacts
          2. Grayscale + Otsu binary threshold — sharpens thin characters
          3. Inverted binary — helps when background is darker than text
        """
        if validator(fields.get(field_name)):
            return fields  # already good

        det = next((d for d in raw_detections if d["class_name"] == field_name), None)
        if det is None:
            logger.debug(f"{field_name} not detected — skipping retry")
            return fields

        crop0 = crop_field(image, det["box_xyxy"])
        if crop0.size == 0:
            return fields

        best_text = fields.get(field_name)
        best_len  = len(best_text) if best_text else 0

        for attempt, retry_crop in enumerate(self._build_retry_crops(crop0), start=1):
            if isinstance(self.ocr, TesseractOCREngine):
                text, conf = self.ocr.read_text_with_confidence(retry_crop, field_name=field_name)
            else:
                text, conf = self.ocr.read_text_with_confidence(retry_crop)

            if conf < self.min_ocr_confidence:
                text = None

            text = normaliser(text)
            logger.debug(f"{field_name} retry {attempt}: '{text}' (conf={conf:.2f})")

            if validator(text):
                logger.info(f"{field_name} corrected on retry {attempt}: {text}")
                fields[field_name] = text
                return fields

            if text and len(text) > best_len:
                best_text = text
                best_len  = len(text)

        if best_text and best_text != fields.get(field_name):
            logger.info(f"{field_name} best partial after retries: {best_text}")
            fields[field_name] = best_text

        return fields

    def _split_job_company(self, crop: np.ndarray):
        """
        Split the Job YOLO crop vertically into two halves:
          top    → Job_Name   (الوظيفة)
          bottom → Company_Name (جهة العمل)

        Returns (job_text, company_text) — either can be None.

        Strategy:
          1. Try newline split from the full-crop OCR result (most reliable).
          2. Fall back to physical top/bottom half if no newline found.
        """
        h, w = crop.shape[:2]

        # ── Attempt 1: multi-line OCR on the full crop ────────────────────────
        if isinstance(self.ocr, TesseractOCREngine):
            full_text, _ = self.ocr.read_text_with_confidence(crop, field_name="Job")
        else:
            full_text, _ = self.ocr.read_text_with_confidence(crop)

        if full_text and "\n" in full_text:
            lines = [l.strip() for l in full_text.split("\n") if l.strip()]
            if len(lines) >= 2:
                return lines[0], lines[1]
            return lines[0], None

        # ── Attempt 2: physical split at midpoint ─────────────────────────────
        # Only worth trying if the crop is tall enough for two lines (>40 px)
        if h < 40:
            return full_text, None

        mid = h // 2
        top_crop    = crop[:mid, :]
        bottom_crop = crop[mid:, :]

        if self.enhance_crops:
            top_crop    = enhance_for_ocr(top_crop,    field_type="text")
            bottom_crop = enhance_for_ocr(bottom_crop, field_type="text")

        if isinstance(self.ocr, TesseractOCREngine):
            job_text,     _ = self.ocr.read_text_with_confidence(top_crop,    field_name="Job")
            company_text, _ = self.ocr.read_text_with_confidence(bottom_crop, field_name="Job")
        else:
            job_text,     _ = self.ocr.read_text_with_confidence(top_crop)
            company_text, _ = self.ocr.read_text_with_confidence(bottom_crop)

        return job_text or None, company_text or None

    def _build_retry_crops(self, crop: np.ndarray) -> List[np.ndarray]:
        """
        Return progressively different preprocessed versions of a crop.
        Used by _retry_field for any field type.
        """
        h, w = crop.shape[:2]
        results = []

        # Strategy 1: raw crop × 3 upscale — no CLAHE/sharpen artefacts
        results.append(cv2.resize(crop, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC))

        # Strategy 2: grayscale + Otsu binary threshold
        gray     = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        gray_big = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))

        # Strategy 3: inverted binary (helps when text is lighter than background)
        results.append(cv2.cvtColor(cv2.bitwise_not(binary), cv2.COLOR_GRAY2RGB))

        return results

    def _identify_side(self, detections: List[Dict]) -> str:
        class_names = {d["class_name"] for d in detections}
        if "Front" in class_names:
            return "front"
        if "Back" in class_names:
            return "back"
        return "unknown"

    @classmethod
    def from_config(cls, config_path: str) -> "ArabicIDOCRPipeline":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        det_cfg = cfg["detection"]
        ocr_cfg = cfg["ocr"]

        detector = FieldDetector(
            model_path=det_cfg.get("model_path", "yolo11n.pt"),
            conf_threshold=det_cfg["conf_threshold"],
            iou_threshold=det_cfg["iou_threshold"],
            img_size=det_cfg["img_size"],
        )

        ocr = create_ocr_engine(
            engine=ocr_cfg["engine"],
            languages=ocr_cfg.get("languages", ["ar", "en"]),
            gpu=ocr_cfg.get("gpu", False),
        )

        return cls(
            detector=detector,
            ocr_engine=ocr,
            min_ocr_confidence=ocr_cfg.get("min_confidence", 0.3),
        )
