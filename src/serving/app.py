"""FastAPI serving application for Arabic ID OCR pipeline."""

import base64
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env before anything reads os.environ (no-op if file doesn't exist)
load_dotenv()

import mlflow
import yaml
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_fastapi_instrumentator import Instrumentator

from src.models.detector import FieldDetector
from src.models.ocr_engine import create_ocr_engine
from src.models.pipeline import ArabicIDOCRPipeline
from src.serving.auth import verify_api_key
from src.serving.schemas import (
    Base64ImageRequest,
    CombinedIDRequest,
    ElsewedyOCRResult,
    ErrorResponse,
    HealthResponse,
    OCRResult,
)

logger = logging.getLogger(__name__)

_pipeline: Optional[ArabicIDOCRPipeline] = None

SERVING_CONFIG       = "configs/serving.yaml"
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024   # 10 MB
SUPPORTED_TYPES      = {"image/jpeg", "image/png", "image/webp"}
MLFLOW_EXPERIMENT    = "arabic-ocr-api"

# ── Egyptian National ID — governorate code lookup ────────────────────────────

GOVERNORATE_CODES: dict = {
    "01": "القاهرة",        "02": "الاسكندرية",    "03": "بورسعيد",
    "04": "السويس",         "11": "دمياط",          "12": "الدقهلية",
    "13": "الشرقية",        "14": "القليوبية",      "15": "كفر الشيخ",
    "16": "الغربية",        "17": "المنوفية",       "18": "البحيرة",
    "19": "الإسماعيلية",    "21": "الجيزة",         "22": "بني سويف",
    "23": "الفيوم",         "24": "المنيا",         "25": "أسيوط",
    "26": "سوهاج",          "27": "قنا",            "28": "أسوان",
    "29": "الأقصر",         "31": "البحر الأحمر",   "32": "الوادي الجديد",
    "33": "مطروح",          "34": "شمال سيناء",     "35": "جنوب سيناء",
    "88": "خارج الجمهورية",
}


def _parse_id_number(id_num: Optional[str]) -> dict:
    """
    Extract structured fields from a 14-digit Egyptian National ID number.

    Format (left to right):
      C(1) + YY(2) + MM(2) + DD(2) + GG(2) + SSSS(4) + D(1) = 14 digits
        C   — century code  (2 = 19xx, 3 = 20xx)
        GG  — governorate code
        SSSS— serial; 4th digit (index 12) odd=Male, even=Female
        D   — check digit
    """
    out: dict = {
        "birth_date": None, "birth_code": None,
        "birth_place_code": None, "governorate": None, "gender": None,
    }
    if not id_num or len(id_num) != 14 or not id_num.isdigit():
        return out

    century_code = id_num[0]
    century = {"2": "19", "3": "20"}.get(century_code)
    if not century:
        return out

    yy, mm, dd = id_num[1:3], id_num[3:5], id_num[5:7]
    gov_code     = id_num[7:9]
    gender_digit = int(id_num[12])

    out["birth_date"]       = f"{dd}/{mm}/{century}{yy}"
    out["birth_code"]       = century_code
    out["birth_place_code"] = gov_code
    out["governorate"]      = GOVERNORATE_CODES.get(gov_code)
    out["gender"]           = "Male" if gender_digit % 2 == 1 else "Female"
    return out


def _split_city_governorate(address2: Optional[str]):
    """
    Try to split 'City - Governorate' from address line 2.
    Returns (city, governorate) — either can be None.
    """
    if not address2:
        return None, None
    if " - " in address2:
        city, gov = address2.split(" - ", 1)
        return city.strip() or None, gov.strip() or None
    return address2.strip() or None, None


def _format_etisalat_response(pipeline_result: dict) -> dict:
    """
    Convert the internal pipeline dict into the Etisalat name/value format.

    Fields that have no equivalent in the current YOLO model (Middle_Name,
    Building_Number, Company_Name) are included with ``value: null`` so the
    consumer always receives a fixed-length property list.
    """
    fields  = pipeline_result.get("fields", {})
    id_num  = fields.get("ID")
    id_info = _parse_id_number(id_num)

    city, gov_from_addr = _split_city_governorate(fields.get("Add2"))
    governorate = gov_from_addr or id_info.get("governorate")

    # Gender: ID-derived is more reliable than OCR of the Arabic vocabulary word
    gender = id_info.get("gender")
    if not gender:
        gender_ar = fields.get("Gender") or ""
        if "ذكر" in gender_ar:
            gender = "Male"
        elif "أنثى" in gender_ar:
            gender = "Female"

    # Split Middle_Name (father's name) from Last_Name (remainder).
    # Special case: "عبد" is a prefix that must be joined with the following word
    # e.g. ["عبد", "الرحمن", "ابوزيد", "على"] → middle="عبدالرحمن", last="ابوزيد على"
    raw_last = (fields.get("Last_Name") or "").strip()
    words     = raw_last.split()

    if not words:
        middle_name = None
        last_name   = None
    elif len(words) == 1:
        # Only one word — no father's name extractable
        middle_name = None
        last_name   = words[0]
    elif words[0] == "عبد" and len(words) >= 2:
        # Standalone "عبد" — combine with next word (no space)
        middle_name = words[0] + words[1]
        last_name   = " ".join(words[2:]) or None
    else:
        middle_name = words[0]
        last_name   = " ".join(words[1:]) or None

    properties = [
        {"name": "ID_Front",         "value": id_num},
        {"name": "First_Name",       "value": fields.get("First_Name")},
        {"name": "Middle_Name",      "value": middle_name},
        {"name": "Last_Name",        "value": last_name},
        {"name": "Building_Number",  "value": None},          # not detected
        {"name": "Address_1",        "value": fields.get("Add1")},
        {"name": "Address_2",        "value": fields.get("Add2")},
        {"name": "Birth_Date",       "value": id_info.get("birth_date")},
        {"name": "City",             "value": city},
        {"name": "Governorate",      "value": governorate},
        {"name": "Birth_Code",       "value": id_info.get("birth_code")},
        {"name": "Birth_Place_Code", "value": id_info.get("birth_place_code")},
        {"name": "Gender",           "value": gender},
        {"name": "ID_Back",          "value": id_num},        # same number on both sides
        {"name": "Job_Name",         "value": fields.get("Job")},
        {"name": "Company_Name",     "value": fields.get("Company")},
        {"name": "Expiry_Date",      "value": fields.get("ExpDate")},
        {"name": "Husband_Name",     "value": fields.get("HusbandName")},
    ]

    return {
        "description": "TEXT_DETECTION fetched data by ElsewedyOCR",
        "properties":  properties,
    }


def _merge_pipeline_fields(front_raw: dict, back_raw: dict) -> dict:
    """
    Merge two pipeline results (front image + back image) into one.

    Strategy:
    - Identity fields (name, ID, gender …) → prefer front
    - Address / job fields              → prefer back
    - Fall back to the other side when the preferred side is null
    """
    front_fields = front_raw.get("fields", {})
    back_fields  = back_raw.get("fields",  {})

    # Fields that live on the back of an Egyptian ID
    back_preferred = {"Job", "Company", "HusbandName", "ExpDate", "IssueDate",
                      "Serial_Num", "Add1", "Add2", "Status", "Religion"}

    all_keys = set(front_fields) | set(back_fields)
    merged: dict = {}
    for key in all_keys:
        fv = front_fields.get(key)
        bv = back_fields.get(key)
        if key in back_preferred:
            merged[key] = bv or fv
        else:
            merged[key] = fv or bv

    total_time = (
        front_raw.get("metadata", {}).get("processing_time_s", 0)
        + back_raw.get("metadata", {}).get("processing_time_s", 0)
    )
    return {
        "fields":   merged,
        "metadata": {
            "num_fields_detected": len([v for v in merged.values() if v]),
            "processing_time_s":   round(total_time, 3),
            "source": "combined",
        },
    }


# CORS — set ALLOWED_ORIGINS env var in production (comma-separated)
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = (
    ["*"] if _raw_origins.strip() == "*"
    else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)


# ── MLflow helpers ────────────────────────────────────────────────────────────

def _setup_mlflow():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:///app/mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        logger.info(f"MLflow tracking URI: {tracking_uri}")
    except Exception as e:
        # MLflow server may not be running yet — app still starts, logging is skipped
        logger.warning(f"MLflow not reachable at startup (non-critical): {e}")


def _log_to_mlflow(result: dict, image_bytes: bytes, source: str = "api"):
    """
    Log an OCR run to MLflow:
      - Metrics  : processing time, field counts
      - Params   : every extracted field value
      - Artifacts: input image  → input_image.jpg
                   output JSON  → output.json
                   output Base64→ output_base64.txt
    """
    try:
        fields   = result.get("fields", {})
        metadata = result.get("metadata", {})
        json_str = json.dumps(result, ensure_ascii=False, indent=2)
        b64_str  = base64.b64encode(json_str.encode("utf-8")).decode("ascii")

        with mlflow.start_run():
            # ── Tags ──────────────────────────────────────────────────────
            mlflow.set_tags({
                "source":  source,
                "id_side": result.get("id_side", "unknown"),
            })

            # ── Metrics ───────────────────────────────────────────────────
            mlflow.log_metrics({
                "processing_time_s":   metadata.get("processing_time_s", 0),
                "num_fields_detected": metadata.get("num_fields_detected", 0),
                "fields_read":         sum(1 for v in fields.values() if v),
                "image_size_bytes":    len(image_bytes),
            })
            field_flags = {f"read_{k}": int(v is not None) for k, v in fields.items()}
            if field_flags:
                mlflow.log_metrics(field_flags)

            # ── Parameters — every extracted field ────────────────────────
            for field_name, value in fields.items():
                mlflow.log_param(field_name, value or "")

            # ── Artifacts ─────────────────────────────────────────────────
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)

                # Input image
                img_path = tmp / "input_image.jpg"
                img_path.write_bytes(image_bytes)
                mlflow.log_artifact(str(img_path), artifact_path="input")

                # Output JSON
                json_path = tmp / "output.json"
                json_path.write_text(json_str, encoding="utf-8")
                mlflow.log_artifact(str(json_path), artifact_path="output")

                # Output Base64
                b64_path = tmp / "output_base64.txt"
                b64_path.write_text(b64_str, encoding="utf-8")
                mlflow.log_artifact(str(b64_path), artifact_path="output")

    except Exception as e:
        logger.warning(f"MLflow logging failed (non-critical): {e}")


# ── Pipeline loader ───────────────────────────────────────────────────────────

def load_pipeline() -> ArabicIDOCRPipeline:
    with open(SERVING_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg     = cfg["model"]
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

    ocr_engine = model_cfg.get("ocr_engine", "paddleocr")
    ocr_gpu    = model_cfg.get("ocr_gpu", False)
    ocr_langs  = model_cfg.get("ocr_languages", ["ar", "en"])

    engine_kwargs: dict = {}
    if ocr_engine == "easyocr":
        engine_kwargs = {"languages": ocr_langs, "gpu": ocr_gpu}
    elif ocr_engine == "paddleocr":
        engine_kwargs = {"lang": "ar", "use_gpu": ocr_gpu}
    elif ocr_engine == "trocr":
        engine_kwargs = {"device": "cpu"}

    logger.info(f"Loading OCR engine: {ocr_engine}")
    ocr = create_ocr_engine(engine=ocr_engine, **engine_kwargs)
    return ArabicIDOCRPipeline(detector=detector, ocr_engine=ocr)


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _setup_mlflow()
    logger.info("Loading Arabic ID OCR pipeline...")
    try:
        _pipeline = load_pipeline()
        logger.info("Pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
    yield
    logger.info("Shutting down")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Arabic ID OCR API",
    description="Extract structured data from Egyptian National ID cards",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Prometheus — exposes /metrics for Prometheus scraping
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    excluded_handlers=["/metrics", "/health"],
).instrument(app).expose(app, include_in_schema=False)


# ── Health (no auth — required for K8s liveness probes) ──────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=_pipeline is not None,
        version="1.0.0",
    )


# ── V1 OCR endpoints (auth required) ─────────────────────────────────────────

@app.post(
    "/v1/ocr",
    response_model=ElsewedyOCRResult,
    summary="Extract fields from an ID card image (multipart upload)",
    tags=["ocr"],
    dependencies=[Depends(verify_api_key)],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def ocr_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="JPEG, PNG, or WebP image of the ID card"),
    include_raw: bool = False,
):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="OCR pipeline not loaded")

    if file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG, PNG, or WebP.",
        )

    data = await file.read()
    if len(data) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Image exceeds 10 MB limit")

    try:
        raw_result = _pipeline.process_bytes(data)
    except Exception as e:
        logger.exception("OCR processing error")
        raise HTTPException(status_code=422, detail=str(e))

    background_tasks.add_task(_log_to_mlflow, raw_result, data, "multipart")

    formatted = _format_etisalat_response(raw_result)
    return Response(
        content=json.dumps(formatted, ensure_ascii=False, indent=2),
        media_type="application/json; charset=utf-8",
    )


@app.post(
    "/v1/ocr/base64",
    response_model=ElsewedyOCRResult,
    summary="Extract fields from a Base64-encoded ID card image",
    tags=["ocr"],
    dependencies=[Depends(verify_api_key)],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def ocr_base64_endpoint(
    request: Base64ImageRequest,
    background_tasks: BackgroundTasks,
):
    """
    Accepts a JSON body with a Base64-encoded image and returns OCR results.

    Example request:
    ```json
    {
        "image": "/9j/4AAQSkZJRgABAQAA...",
        "include_raw": false
    }
    ```
    Supports optional data URI prefix: ``data:image/jpeg;base64,...``
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="OCR pipeline not loaded")

    try:
        image_bytes = base64.b64decode(request.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to decode Base64 image.")

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Image exceeds 10 MB limit")

    try:
        raw_result = _pipeline.process_bytes(image_bytes)
    except Exception as e:
        logger.exception("OCR processing error")
        raise HTTPException(status_code=422, detail=str(e))

    background_tasks.add_task(_log_to_mlflow, raw_result, image_bytes, "base64")

    formatted = _format_etisalat_response(raw_result)
    return Response(
        content=json.dumps(formatted, ensure_ascii=False, indent=2),
        media_type="application/json; charset=utf-8",
    )


@app.post(
    "/v1/ocr/combined",
    response_model=ElsewedyOCRResult,
    summary="Process front AND back of an ID card in one call",
    description=(
        "Accepts Base64-encoded front and back images, runs OCR on each, "
        "then merges the results into a single ElsewedyOCR JSON. "
        "Identity fields (name, gender, ID) come from the front; "
        "address and job fields come from the back."
    ),
    tags=["ocr"],
    dependencies=[Depends(verify_api_key)],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def ocr_combined_endpoint(
    request: CombinedIDRequest,
    background_tasks: BackgroundTasks,
):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="OCR pipeline not loaded")

    try:
        front_bytes = base64.b64decode(request.front)
        back_bytes  = base64.b64decode(request.back)
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to decode one of the Base64 images.")

    for img_bytes in (front_bytes, back_bytes):
        if len(img_bytes) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(status_code=400, detail="One or more images exceed the 10 MB limit.")

    try:
        front_raw = _pipeline.process_bytes(front_bytes)
        back_raw  = _pipeline.process_bytes(back_bytes)
    except Exception as e:
        logger.exception("OCR processing error (combined)")
        raise HTTPException(status_code=422, detail=str(e))

    merged_raw = _merge_pipeline_fields(front_raw, back_raw)
    background_tasks.add_task(_log_to_mlflow, merged_raw, front_bytes, "combined")

    formatted = _format_etisalat_response(merged_raw)
    return Response(
        content=json.dumps(formatted, ensure_ascii=False, indent=2),
        media_type="application/json; charset=utf-8",
    )


@app.post(
    "/v1/ocr/batch",
    summary="Process multiple ID card images (max 10)",
    tags=["ocr"],
    dependencies=[Depends(verify_api_key)],
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def ocr_batch(
    background_tasks: BackgroundTasks,
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
            raw = _pipeline.process_bytes(data)
            background_tasks.add_task(_log_to_mlflow, raw, data, "batch")
            results.append({"filename": f.filename, "result": _format_etisalat_response(raw)})
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    return Response(
        content=json.dumps(results, ensure_ascii=False, indent=2),
        media_type="application/json; charset=utf-8",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.app:app", host="0.0.0.0", port=8000, reload=True)
