"""Arabic OCR engine supporting EasyOCR and PaddleOCR backends."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BaseOCREngine(ABC):
    @abstractmethod
    def read_text(self, image: np.ndarray) -> str:
        pass

    @abstractmethod
    def read_text_with_confidence(self, image: np.ndarray) -> Tuple[str, float]:
        pass


class EasyOCREngine(BaseOCREngine):
    """EasyOCR backend — supports Arabic + English out of the box."""

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False):
        try:
            import easyocr
        except ImportError:
            raise ImportError("Install easyocr: pip install easyocr")

        self.languages = languages or ["ar", "en"]
        self.reader = easyocr.Reader(self.languages, gpu=gpu, verbose=False)

    def read_text(self, image: np.ndarray) -> str:
        text, _ = self.read_text_with_confidence(image)
        return text

    def read_text_with_confidence(self, image: np.ndarray) -> Tuple[str, float]:
        results = self.reader.readtext(image, detail=1, paragraph=False)
        if not results:
            return "", 0.0

        texts, confs = [], []
        for _, text, conf in results:
            if text.strip():
                texts.append(text.strip())
                confs.append(conf)

        combined = " ".join(texts)
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return combined, round(avg_conf, 4)


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR backend — stronger Arabic model, heavier dependency."""

    def __init__(self, lang: str = "arabic", use_gpu: bool = False):
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError("Install paddleocr: pip install paddlepaddle paddleocr")

        self.ocr = PaddleOCR(lang=lang, use_gpu=use_gpu, show_log=False)

    def read_text(self, image: np.ndarray) -> str:
        text, _ = self.read_text_with_confidence(image)
        return text

    def read_text_with_confidence(self, image: np.ndarray) -> Tuple[str, float]:
        result = self.ocr.ocr(image, cls=True)
        if not result or not result[0]:
            return "", 0.0

        texts, confs = [], []
        for line in result[0]:
            text, conf = line[1]
            if text.strip():
                texts.append(text.strip())
                confs.append(conf)

        combined = " ".join(texts)
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return combined, round(avg_conf, 4)


def create_ocr_engine(engine: str = "easyocr", **kwargs) -> BaseOCREngine:
    engines = {
        "easyocr": EasyOCREngine,
        "paddleocr": PaddleOCREngine,
    }
    if engine not in engines:
        raise ValueError(f"Unknown OCR engine '{engine}'. Choose from: {list(engines)}")
    return engines[engine](**kwargs)
