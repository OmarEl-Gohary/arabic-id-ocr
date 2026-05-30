"""End-to-end pipeline: image → field detection → OCR → JSON."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from src.data.preprocess import bytes_to_image, crop_field, enhance_for_ocr, load_image
from src.models.detector import FieldDetector
from src.models.ocr_engine import BaseOCREngine, create_ocr_engine
from src.models.postprocess import postprocess_fields

logger = logging.getLogger(__name__)

STRUCTURAL_FIELDS = {"Front", "Back"}
NUMERIC_FIELDS = {"ID", "Serial_Num", "ExpDate", "IssueDate"}


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
                crop = enhance_for_ocr(crop)

            text, conf = self.ocr.read_text_with_confidence(crop)

            if conf < self.min_ocr_confidence:
                text = None

            fields[field_name] = text
            raw_detections.append({
                **det,
                "ocr_text": text,
                "ocr_confidence": conf,
            })

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
