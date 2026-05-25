"""Data validation for the Arabic OCR dataset and predictions."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


EXPECTED_CLASSES = {
    "Add1", "Add2", "Back", "ExpDate", "First_Name", "Front",
    "Gender", "HusbandName", "ID", "IssueDate", "Job",
    "Last_Name", "Religion", "Serial_Num", "Status"
}

FRONT_REQUIRED_FIELDS = {"First_Name", "Last_Name", "ID", "Gender", "Religion"}
BACK_REQUIRED_FIELDS = {"Add1", "Serial_Num", "IssueDate", "ExpDate"}


class DatasetValidator:
    """Validates dataset integrity before training."""

    def __init__(self, dataset_root: str):
        self.root = Path(dataset_root)
        self.issues: List[str] = []
        self.stats: Dict = {}

    def run(self) -> bool:
        self._check_structure()
        self._check_yaml()
        self._check_annotations()
        return len(self.issues) == 0

    def _check_structure(self):
        for split in ["train", "valid", "test"]:
            for sub in ["images", "labels"]:
                d = self.root / split / sub
                if not d.exists():
                    self.issues.append(f"Missing directory: {d}")

    def _check_yaml(self):
        yaml_path = self.root / "data.yaml"
        if not yaml_path.exists():
            self.issues.append("data.yaml not found")
            return
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        if set(cfg.get("names", [])) != EXPECTED_CLASSES:
            self.issues.append(f"Class mismatch in data.yaml: {cfg.get('names')}")
        if cfg.get("nc") != 15:
            self.issues.append(f"Expected 15 classes, got {cfg.get('nc')}")

    def _check_annotations(self):
        counts = {}
        for split in ["train", "valid", "test"]:
            lbl_dir = self.root / split / "labels"
            img_dir = self.root / split / "images"
            if not lbl_dir.exists():
                continue
            n_images = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
            n_labels = len(list(lbl_dir.glob("*.txt")))
            counts[split] = {"images": n_images, "labels": n_labels}

            for label_file in lbl_dir.glob("*.txt"):
                with open(label_file) as f:
                    for lineno, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if not parts:
                            continue
                        # valid formats: bbox (5) or polygon (1 + 2*n coords, n>=4)
                        n = len(parts)
                        is_bbox = n == 5
                        is_polygon = n >= 9 and (n - 1) % 2 == 0
                        if not (is_bbox or is_polygon):
                            self.issues.append(
                                f"{label_file.name} line {lineno}: unexpected value count {n}"
                            )
                            continue
                        cls_id = int(parts[0])
                        if cls_id < 0 or cls_id >= 15:
                            self.issues.append(
                                f"{label_file.name} line {lineno}: invalid class id {cls_id}"
                            )
                        coords = list(map(float, parts[1:]))
                        if any(c < 0 or c > 1 for c in coords):
                            self.issues.append(
                                f"{label_file.name} line {lineno}: coords out of [0,1] range"
                            )
        self.stats["split_counts"] = counts

    def report(self) -> Dict:
        return {
            "valid": len(self.issues) == 0,
            "issues": self.issues,
            "stats": self.stats,
        }


class PredictionValidator:
    """Validates model prediction outputs."""

    def validate(self, prediction: Dict[str, Any]) -> List[str]:
        errors = []
        if not isinstance(prediction, dict):
            return ["Prediction must be a dict"]
        if "fields" not in prediction:
            errors.append("Missing 'fields' key")
        else:
            for field_name, value in prediction["fields"].items():
                if field_name not in EXPECTED_CLASSES:
                    errors.append(f"Unknown field: {field_name}")
                if not isinstance(value, (str, type(None))):
                    errors.append(f"Field {field_name} value must be str or null")
        return errors

    def validate_batch(self, predictions: List[Dict]) -> Dict:
        total = len(predictions)
        error_counts = 0
        all_errors = []
        for i, pred in enumerate(predictions):
            errs = self.validate(pred)
            if errs:
                error_counts += 1
                all_errors.extend([f"[{i}] {e}" for e in errs])
        return {
            "total": total,
            "invalid": error_counts,
            "error_rate": error_counts / total if total else 0,
            "errors": all_errors,
        }
