"""Dataset utilities for Egyptian ID card field detection."""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from torch.utils.data import Dataset


FIELD_CLASSES = [
    "Add1", "Add2", "Back", "ExpDate", "First_Name", "Front",
    "Gender", "HusbandName", "ID", "IssueDate", "Job",
    "Last_Name", "Religion", "Serial_Num", "Status"
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(FIELD_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}


class EgyptianIDDataset(Dataset):
    """YOLOv11-format dataset for Egyptian ID card field detection."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 640,
        transforms=None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.transforms = transforms

        self.images_dir = self.data_dir / split / "images"
        self.labels_dir = self.data_dir / split / "labels"

        self.image_paths = sorted(list(self.images_dir.glob("*.jpg")) +
                                  list(self.images_dir.glob("*.png")))

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.image_paths[idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, class_ids = self._load_yolo_labels(label_path)

        if self.transforms:
            image = self.transforms(image)

        return {
            "image": image,
            "boxes": boxes,
            "class_ids": class_ids,
            "image_path": str(img_path),
        }

    def _load_yolo_labels(self, label_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        boxes, class_ids = [], []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        boxes.append([cx, cy, w, h])
                        class_ids.append(cls_id)
        return np.array(boxes, dtype=np.float32), np.array(class_ids, dtype=np.int64)

    def get_class_distribution(self) -> Dict[str, int]:
        dist = {cls: 0 for cls in FIELD_CLASSES}
        for img_path in self.image_paths:
            label_path = self.labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_name = IDX_TO_CLASS.get(int(parts[0]))
                            if cls_name:
                                dist[cls_name] += 1
        return dist

    def get_split_stats(self) -> Dict:
        dist = self.get_class_distribution()
        return {
            "split": self.split,
            "total_images": len(self.image_paths),
            "total_annotations": sum(dist.values()),
            "class_distribution": dist,
        }


def load_data_config(config_path: str) -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_dataset_structure(dataset_root: str) -> List[str]:
    """Return list of issues found; empty list means OK."""
    root = Path(dataset_root)
    issues = []

    for split in ["train", "valid", "test"]:
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.exists():
            issues.append(f"Missing {img_dir}")
        if not lbl_dir.exists():
            issues.append(f"Missing {lbl_dir}")

    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        issues.append(f"Missing {data_yaml}")

    if not issues:
        for split in ["train", "valid", "test"]:
            img_dir = root / split / "images"
            lbl_dir = root / split / "labels"
            images = set(p.stem for p in img_dir.glob("*.jpg")) | set(p.stem for p in img_dir.glob("*.png"))
            labels = set(p.stem for p in lbl_dir.glob("*.txt"))
            unlabeled = images - labels
            if unlabeled:
                issues.append(f"{split}: {len(unlabeled)} images without labels")

    return issues
