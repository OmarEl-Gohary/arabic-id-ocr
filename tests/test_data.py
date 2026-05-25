"""Tests for data loading and validation."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.dataset import (
    CLASS_TO_IDX,
    FIELD_CLASSES,
    IDX_TO_CLASS,
    validate_dataset_structure,
)
from src.data.preprocess import (
    bytes_to_image,
    crop_field,
    enhance_for_ocr,
    resize_with_padding,
)
from src.data.validate import DatasetValidator, PredictionValidator


def make_dummy_image(h=480, w=640) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestFieldClasses:
    def test_class_count(self):
        assert len(FIELD_CLASSES) == 15

    def test_class_to_idx_roundtrip(self):
        for cls in FIELD_CLASSES:
            assert IDX_TO_CLASS[CLASS_TO_IDX[cls]] == cls

    def test_expected_classes_present(self):
        for name in ["First_Name", "Last_Name", "ID", "Front", "Back"]:
            assert name in FIELD_CLASSES


class TestPreprocessing:
    def test_resize_with_padding_shape(self):
        img = make_dummy_image(480, 640)
        result, scale, (pw, ph) = resize_with_padding(img, 640)
        assert result.shape == (640, 640, 3)
        assert 0 < scale <= 1.0

    def test_resize_square_input(self):
        img = make_dummy_image(640, 640)
        result, scale, _ = resize_with_padding(img, 640)
        assert result.shape == (640, 640, 3)
        assert scale == 1.0

    def test_crop_field_valid(self):
        img = make_dummy_image(480, 640)
        crop = crop_field(img, (100, 50, 300, 150))
        assert crop.shape[0] > 0 and crop.shape[1] > 0

    def test_crop_field_with_padding(self):
        img = make_dummy_image(480, 640)
        crop = crop_field(img, (100, 50, 300, 150), padding=8)
        assert crop.shape[0] > 0

    def test_enhance_for_ocr_output_shape(self):
        img = make_dummy_image(100, 200)
        enhanced = enhance_for_ocr(img)
        assert enhanced.shape == img.shape

    def test_bytes_to_image_invalid_raises(self):
        with pytest.raises(ValueError):
            bytes_to_image(b"not an image")


class TestPredictionValidator:
    def setup_method(self):
        self.validator = PredictionValidator()

    def test_valid_prediction(self):
        pred = {
            "fields": {
                "First_Name": "أحمد",
                "Last_Name": "محمد",
                "ID": "29901011234567",
            }
        }
        errors = self.validator.validate(pred)
        assert errors == []

    def test_missing_fields_key(self):
        errors = self.validator.validate({"id_side": "front"})
        assert any("fields" in e for e in errors)

    def test_unknown_field_name(self):
        pred = {"fields": {"UnknownField": "value"}}
        errors = self.validator.validate(pred)
        assert any("Unknown field" in e for e in errors)

    def test_null_field_value_is_valid(self):
        pred = {"fields": {"First_Name": None}}
        errors = self.validator.validate(pred)
        assert errors == []


class TestDatasetStructure:
    def test_validate_missing_dirs(self, tmp_path):
        issues = validate_dataset_structure(str(tmp_path))
        assert len(issues) > 0
        assert any("train" in i for i in issues)

    def test_validate_ok_structure(self, tmp_path):
        for split in ["train", "valid", "test"]:
            (tmp_path / split / "images").mkdir(parents=True)
            (tmp_path / split / "labels").mkdir(parents=True)
        # Write minimal data.yaml
        import yaml
        with open(tmp_path / "data.yaml", "w") as f:
            yaml.dump({"nc": 15, "names": [
                "Add1", "Add2", "Back", "ExpDate", "First_Name", "Front",
                "Gender", "HusbandName", "ID", "IssueDate", "Job",
                "Last_Name", "Religion", "Serial_Num", "Status"
            ]}, f)
        issues = validate_dataset_structure(str(tmp_path))
        assert issues == []
