"""FastAPI serving application for Arabic ID OCR pipeline."""

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from src.models.detector import FieldDetector
from src.models.ocr_engine import create_ocr_engine
from src.models.pipeline import ArabicIDOCRPipeline
from src.serving.schemas import ErrorResponse, HealthResponse, OCRResult

logger = logging.getLogger(__name__)

_pipeline: Optional[ArabicIDOCRPipeline] = None

SERVING_CONFIG = "configs/serving.yaml"
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def load_pipeline() -> ArabicIDOCRPipeline:
    with open(SERVING_CONFIG) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    detector_path = model_cfg["detector_path"]

    if not Path(detector_path).exists():
        logger.warning(f"Detector weights not found at {detector_path}, using base model")
        detector_path = "yolo11n.pt"

    detector = FieldDetector(
        model_path=detector_path,
        conf_threshold=0.25,
        iou_threshold=0.45,
        device=model_cfg.get("device", "cpu"),
    )
    ocr_engine  = model_cfg.get("ocr_engine", "easyocr")
    ocr_gpu     = model_cfg.get("ocr_gpu", False)
    ocr_langs   = model_cfg.get("ocr_languages", ["ar", "en"])

    engine_kwargs = {}
    if ocr_engine == "easyocr":
        engine_kwargs = {"languages": ocr_langs, "gpu": ocr_gpu}
    elif ocr_engine == "paddleocr":
        engine_kwargs = {"lang": "arabic", "use_gpu": ocr_gpu}
    elif ocr_engine == "trocr":
        engine_kwargs = {"device": "cpu"}

    logger.info(f"Loading OCR engine: {ocr_engine}")
    ocr = create_ocr_engine(engine=ocr_engine, **engine_kwargs)
    return ArabicIDOCRPipeline(detector=detector, ocr_engine=ocr)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    logger.info("Loading Arabic ID OCR pipeline...")
    try:
        _pipeline = load_pipeline()
        logger.info("Pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Arabic ID OCR API",
    description="Extract structured data from Egyptian National ID cards",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=_pipeline is not None,
    )


@app.post(
    "/ocr",
    response_model=OCRResult,
    summary="Extract fields from an ID card image",
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def ocr_endpoint(
    file: UploadFile = File(..., description="JPEG, PNG, or WebP image of the ID card"),
    include_raw: bool = False,
):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="OCR pipeline not loaded")

    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG, PNG, or WebP.",
        )

    data = await file.read()
    if len(data) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Image exceeds 10 MB limit")

    try:
        result = _pipeline.process_bytes(data)
    except Exception as e:
        logger.exception("OCR processing error")
        raise HTTPException(status_code=422, detail=str(e))

    if not include_raw:
        result.pop("raw_detections", None)

    return JSONResponse(content=result)


@app.post(
    "/ocr/batch",
    summary="Process multiple ID card images",
    response_model=list,
)
async def ocr_batch(
    files: list[UploadFile] = File(...),
):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="OCR pipeline not loaded")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")

    results = []
    for f in files:
        data = await f.read()
        try:
            res = _pipeline.process_bytes(data)
            res.pop("raw_detections", None)
            results.append({"filename": f.filename, "result": res})
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.app:app", host="0.0.0.0", port=8000, reload=True)
