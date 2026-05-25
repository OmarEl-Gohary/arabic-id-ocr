"""Tests for model components (detector, OCR, pipeline)."""

from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.pipeline import ArabicIDOCRPipeline, STRUCTURAL_FIELDS


def make_test_image(h=640, w=640) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def make_mock_detector(detections: List[Dict]) -> MagicMock:
    detector = MagicMock()
    detector.detect.return_value = detections
    return detector


def make_mock_ocr(text: str = "أحمد", conf: float = 0.85) -> MagicMock:
    ocr = MagicMock()
    ocr.read_text_with_confidence.return_value = (text, conf)
    return ocr


class TestArabicIDOCRPipeline:
    def _make_pipeline(self, detections):
        detector = make_mock_detector(detections)
        ocr = make_mock_ocr()
        return ArabicIDOCRPipeline(
            detector=detector,
            ocr_engine=ocr,
            min_ocr_confidence=0.3,
            enhance_crops=False,
        )

    def test_process_image_returns_fields(self):
        detections = [
            {"class_name": "First_Name", "confidence": 0.9, "box_xyxy": [10, 10, 100, 40]},
            {"class_name": "Last_Name", "confidence": 0.85, "box_xyxy": [10, 50, 100, 80]},
        ]
        pipeline = self._make_pipeline(detections)
        result = pipeline.process_image(make_test_image())

        assert "fields" in result
        assert "First_Name" in result["fields"]
        assert "Last_Name" in result["fields"]

    def test_structural_fields_excluded_from_ocr(self):
        detections = [
            {"class_name": "Front", "confidence": 0.95, "box_xyxy": [0, 0, 640, 640]},
            {"class_name": "ID", "confidence": 0.9, "box_xyxy": [50, 100, 200, 130]},
        ]
        pipeline = self._make_pipeline(detections)
        result = pipeline.process_image(make_test_image())

        assert "Front" not in result["fields"]
        assert result["id_side"] == "front"

    def test_id_side_detection_front(self):
        detections = [{"class_name": "Front", "confidence": 0.9, "box_xyxy": [0, 0, 640, 640]}]
        pipeline = self._make_pipeline(detections)
        result = pipeline.process_image(make_test_image())
        assert result["id_side"] == "front"

    def test_id_side_detection_back(self):
        detections = [{"class_name": "Back", "confidence": 0.9, "box_xyxy": [0, 0, 640, 640]}]
        pipeline = self._make_pipeline(detections)
        result = pipeline.process_image(make_test_image())
        assert result["id_side"] == "back"

    def test_low_confidence_ocr_returns_none(self):
        detections = [
            {"class_name": "First_Name", "confidence": 0.9, "box_xyxy": [10, 10, 100, 40]},
        ]
        detector = make_mock_detector(detections)
        ocr = MagicMock()
        ocr.read_text_with_confidence.return_value = ("", 0.1)  # below threshold

        pipeline = ArabicIDOCRPipeline(
            detector=detector,
            ocr_engine=ocr,
            min_ocr_confidence=0.3,
            enhance_crops=False,
        )
        result = pipeline.process_image(make_test_image())
        assert result["fields"]["First_Name"] is None

    def test_metadata_contains_timing(self):
        pipeline = self._make_pipeline([])
        result = pipeline.process_image(make_test_image())
        assert "processing_time_s" in result["metadata"]
        assert result["metadata"]["processing_time_s"] >= 0

    def test_empty_detection_returns_empty_fields(self):
        pipeline = self._make_pipeline([])
        result = pipeline.process_image(make_test_image())
        assert result["fields"] == {}
        assert result["id_side"] == "unknown"
