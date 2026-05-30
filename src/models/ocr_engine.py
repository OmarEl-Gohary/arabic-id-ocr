"""Arabic OCR engine — supports EasyOCR, PaddleOCR, and TrOCR backends."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BaseOCREngine(ABC):
    @abstractmethod
    def read_text(self, image: np.ndarray) -> str:
        pass

    @abstractmethod
    def read_text_with_confidence(self, image: np.ndarray) -> Tuple[str, float]:
        pass


# ── EasyOCR ───────────────────────────────────────────────────────────────────

class EasyOCREngine(BaseOCREngine):
    """EasyOCR backend — supports Arabic + English out of the box."""

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False):
        try:
            import easyocr
        except ImportError:
            raise ImportError("pip install easyocr")

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


# ── PaddleOCR ─────────────────────────────────────────────────────────────────

class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR — stronger Arabic model, better at dense text."""

    def __init__(self, lang: str = "arabic", use_gpu: bool = False):
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError("pip install paddlepaddle paddleocr")

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


# ── TrOCR ─────────────────────────────────────────────────────────────────────

class TrOCREngine(BaseOCREngine):
    """
    Microsoft TrOCR — transformer encoder-decoder, best for printed text.

    Uses a fine-tuned Arabic model from HuggingFace by default.
    Falls back to the base printed model if the Arabic one is unavailable.

    Install: pip install transformers torch Pillow
    """

    DEFAULT_ARABIC_MODEL = "EkberJafar/trocr-base-arabic-v1"
    FALLBACK_MODEL       = "microsoft/trocr-base-printed"

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            from PIL import Image as PILImage
        except ImportError:
            raise ImportError("pip install transformers Pillow")

        import torch
        self.torch  = torch
        self.PILImage = PILImage
        self.device = device

        model_id = model_name or self.DEFAULT_ARABIC_MODEL
        logger.info(f"Loading TrOCR model: {model_id}")

        try:
            self.processor = TrOCRProcessor.from_pretrained(model_id)
            self.model     = VisionEncoderDecoderModel.from_pretrained(model_id)
        except Exception as e:
            logger.warning(f"Could not load {model_id}: {e}. Falling back to {self.FALLBACK_MODEL}")
            self.processor = TrOCRProcessor.from_pretrained(self.FALLBACK_MODEL)
            self.model     = VisionEncoderDecoderModel.from_pretrained(self.FALLBACK_MODEL)

        self.model.to(device)
        self.model.eval()

    def read_text(self, image: np.ndarray) -> str:
        text, _ = self.read_text_with_confidence(image)
        return text

    def read_text_with_confidence(self, image: np.ndarray) -> Tuple[str, float]:
        import torch

        # Convert numpy RGB → PIL
        pil_img = self.PILImage.fromarray(image).convert("RGB")

        pixel_values = self.processor(
            images=pil_img, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)

        text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        # TrOCR doesn't expose a confidence score natively;
        # return 0.9 as a placeholder when text is found
        conf = 0.9 if text else 0.0
        return text, conf


# ── Factory ───────────────────────────────────────────────────────────────────

def create_ocr_engine(engine: str = "easyocr", **kwargs) -> BaseOCREngine:
    """
    Create an OCR engine by name.

    Args:
        engine: "easyocr" | "paddleocr" | "trocr"
        **kwargs: passed to the engine constructor

    Examples:
        create_ocr_engine("easyocr", languages=["ar","en"], gpu=False)
        create_ocr_engine("paddleocr", lang="arabic")
        create_ocr_engine("trocr", model_name="EkberJafar/trocr-base-arabic-v1")
    """
    engines = {
        "easyocr":   EasyOCREngine,
        "paddleocr": PaddleOCREngine,
        "trocr":     TrOCREngine,
    }
    if engine not in engines:
        raise ValueError(f"Unknown OCR engine '{engine}'. Choose from: {list(engines)}")
    logger.info(f"Initialising OCR engine: {engine}")
    return engines[engine](**kwargs)
