"""Arabic OCR engine — supports Tesseract, EasyOCR, PaddleOCR, and TrOCR backends."""

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


# ── Tesseract ─────────────────────────────────────────────────────────────────

class TesseractOCREngine(BaseOCREngine):
    """
    Tesseract — lightest option, ideal for on-prem / limited resources.
    Runs on CPU only, ~100MB RAM, handles printed Arabic well.

    Install:
        macOS:  brew install tesseract tesseract-lang
        Ubuntu: apt install tesseract-ocr tesseract-ocr-ara
        pip:    pip install pytesseract
    """

    def __init__(self, lang: str = "ara+eng", config: str = ""):
        try:
            import pytesseract
        except ImportError:
            raise ImportError("pip install pytesseract  (also: brew install tesseract tesseract-lang)")

        import pytesseract as tess
        self.tess = tess
        # ara = Arabic, eng = English fallback for numbers/serial
        self.lang   = lang
        # PSM 6: assume uniform block of text — best for single cropped fields
        self.config = config or "--psm 6 --oem 1"

    def read_text(self, image: np.ndarray) -> str:
        text, _ = self.read_text_with_confidence(image)
        return text

    # Fields where a single text line is expected → PSM 7 works best
    _SINGLE_LINE_FIELDS = {
        "ID", "Serial_Num", "ExpDate", "IssueDate",
        "First_Name", "Last_Name", "HusbandName",
        "Gender", "Religion", "Job", "Status",
    }
    # Multi-line fields (address blocks) → PSM 6
    _BLOCK_FIELDS = {"Add1", "Add2"}

    # ── Field → (lang, whitelist, psm_candidates) ─────────────────────────────
    #
    # Numeric-only fields: use eng + digit whitelist — far more reliable than
    # asking the Arabic model to read digits on a coloured ID card background.
    #
    # Arabic text fields: use 'ara' alone — adding 'eng' causes the engine to
    # hallucinate Latin characters.
    #
    # PSM 7 = single text line, PSM 6 = uniform block of text.

    # Arabic-Indic digits (٠-٩) + ASCII digits (0-9) for numeric fields.
    # Egyptian ID cards print the national ID number and dates in Arabic-Indic
    # numerals, so we must use the 'ara' language model (which knows ٠-٩) and
    # whitelist both digit sets so Latin-looking noise is ignored.
    _ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
    _DIGIT_WHITELIST      = "٠١٢٣٤٥٦٧٨٩0123456789"
    _DATE_WHITELIST       = "٠١٢٣٤٥٦٧٨٩0123456789/-."

    _FIELD_CFG: dict = {
        # (lang,  whitelist,       psm_candidates)
        "ID":          ("ara", _DIGIT_WHITELIST, [7]),
        "Serial_Num":  ("ara", _DIGIT_WHITELIST, [7]),
        "ExpDate":     ("ara", _DATE_WHITELIST,  [7]),
        "IssueDate":   ("ara", _DATE_WHITELIST,  [7]),
        # Arabic-only text fields — no whitelist so all Arabic chars are accepted
        "First_Name":  ("ara", "",               [7, 6]),
        "Last_Name":   ("ara", "",               [7, 6]),
        "HusbandName": ("ara", "",               [7, 6]),
        "Gender":      ("ara", "",               [7]),
        "Religion":    ("ara", "",               [7]),
        "Job":         ("ara", "",               [7, 6]),
        "Status":      ("ara", "",               [6, 7]),
        "Add1":        ("ara", "",               [6]),
        "Add2":        ("ara", "",               [6]),
    }
    _DEFAULT_CFG = ("ara+eng", "", [6, 7])

    def _run_tess(
        self,
        binary: np.ndarray,
        psm: int,
        lang: str,
        whitelist: str,
    ) -> Tuple[str, float]:
        """Run Tesseract with a given PSM/lang/whitelist and return (text, conf)."""
        wl_flag = f" -c tessedit_char_whitelist={whitelist}" if whitelist else ""
        cfg = f"--psm {psm} --oem 1{wl_flag}"
        text = self.tess.image_to_string(binary, lang=lang, config=cfg).strip()
        data = self.tess.image_to_data(
            binary, lang=lang, config=cfg,
            output_type=self.tess.Output.DICT,
        )
        confs = [c for c in data["conf"] if isinstance(c, (int, float)) and c >= 0]
        avg_conf = float(np.mean(confs)) / 100.0 if confs else 0.0
        return text, round(avg_conf, 4)

    def read_text_with_confidence(
        self,
        image: np.ndarray,
        field_name: Optional[str] = None,
    ) -> Tuple[str, float]:
        import cv2

        # ── Convert to grayscale ──────────────────────────────────────────────
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        # ── Field-specific Tesseract config ───────────────────────────────────
        lang, whitelist, psm_candidates = self._FIELD_CFG.get(
            field_name, self._DEFAULT_CFG
        )

        # ── Build binarized candidates ────────────────────────────────────────
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Adaptive threshold handles uneven lighting / tinted ID backgrounds
        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=15, C=8,
        )

        # ── Try (binary × psm) combos; keep highest confidence ───────────────
        best_text, best_conf = "", 0.0
        for binary in (otsu, adaptive):
            for psm in psm_candidates:
                t, c = self._run_tess(binary, psm, lang, whitelist)
                if c > best_conf or (c == best_conf and len(t) > len(best_text)):
                    best_text, best_conf = t, c
                if best_conf >= 0.75:           # good enough — stop early
                    return best_text, best_conf

        return best_text, best_conf


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
    """
    PaddleOCR v3 — uses TextRecognition only (no internal text detection).

    YOLO already localised each field crop, so we skip PaddleOCR's built-in
    detector and run only the Arabic recognition model directly on the crop.
    Faster and more accurate on small pre-cropped images.
    """

    def __init__(self, lang: str = "ar", use_gpu: bool = False):
        try:
            from paddleocr import TextRecognition
        except ImportError:
            raise ImportError("pip install paddlepaddle paddleocr>=3.0")

        self.rec  = TextRecognition(model_name="arabic_PP-OCRv5_mobile_rec")
        self.lang = lang

    def read_text(self, image: np.ndarray) -> str:
        text, _ = self.read_text_with_confidence(image)
        return text

    def read_text_with_confidence(self, image: np.ndarray) -> Tuple[str, float]:
        import cv2
        # TextRecognition requires BGR numpy array
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image
        result = self.rec.predict(bgr)

        if not result:
            return "", 0.0

        texts, confs = [], []
        for item in result:
            if hasattr(item, "get") or isinstance(item, dict):
                t = item.get("rec_text", "") or ""
                c = item.get("rec_score", 0.0) or 0.0
                if t.strip():
                    texts.append(t.strip())
                    confs.append(float(c))

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

    DEFAULT_ARABIC_MODEL = "microsoft/trocr-base-printed"   # fine-tuned model path goes here once trained
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
        "tesseract": TesseractOCREngine,
        "easyocr":   EasyOCREngine,
        "paddleocr": PaddleOCREngine,
        "trocr":     TrOCREngine,
    }
    if engine not in engines:
        raise ValueError(f"Unknown OCR engine '{engine}'. Choose from: {list(engines)}")
    logger.info(f"Initialising OCR engine: {engine}")
    return engines[engine](**kwargs)
