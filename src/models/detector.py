"""YOLOv11 field detector wrapper for Egyptian ID cards."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

from src.data.dataset import IDX_TO_CLASS


class FieldDetector:
    """Wraps a YOLO model to detect fields on Egyptian ID cards."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        img_size: int = 640,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Run detection on a single RGB image.
        Returns list of dicts with keys: class_name, confidence, box_xyxy.
        """
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                detections.append({
                    "class_name": IDX_TO_CLASS.get(cls_id, f"class_{cls_id}"),
                    "class_id": cls_id,
                    "confidence": round(conf, 4),
                    "box_xyxy": [round(v, 1) for v in xyxy],
                })

        return self._deduplicate(detections)

    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """Run detection on a batch of images."""
        return [self.detect(img) for img in images]

    def _deduplicate(self, detections: List[Dict]) -> List[Dict]:
        """Keep highest-confidence detection per class (IDs have one of each field)."""
        best: Dict[str, Dict] = {}
        for det in detections:
            cls = det["class_name"]
            if cls not in best or det["confidence"] > best[cls]["confidence"]:
                best[cls] = det
        return sorted(best.values(), key=lambda d: d["box_xyxy"][1])  # sort top-to-bottom

    @classmethod
    def from_pretrained(cls, model_name: str = "yolo11n.pt", **kwargs) -> "FieldDetector":
        """Load a pretrained base model (fine-tune before use on ID cards)."""
        return cls(model_path=model_name, **kwargs)
