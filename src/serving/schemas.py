"""Pydantic schemas for the OCR API request/response."""

from typing import Any, Dict, List, Optional

import base64
from pydantic import BaseModel, Field, field_validator


class DetectionBox(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    box_xyxy: List[float]
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None


class PipelineMetadata(BaseModel):
    num_fields_detected: int
    processing_time_s: float
    source: Optional[str] = None


class OCRResult(BaseModel):
    id_side: str = Field(..., description="'front', 'back', or 'unknown'")
    fields: Dict[str, Optional[str]] = Field(
        ...,
        description="Extracted field values keyed by field name",
        example={
            "First_Name": "أحمد",
            "Last_Name": "محمد",
            "ID": "29901011234567",
            "Gender": "ذكر",
            "Religion": "مسلم",
            "ExpDate": "01/01/2030",
            "IssueDate": "01/01/2020",
            "Serial_Num": "12345",
        }
    )
    metadata: PipelineMetadata
    output_json: Optional[str] = Field(None, description="Full result as a JSON string")
    output_base64: Optional[str] = Field(None, description="Full result encoded as Base64")
    raw_detections: Optional[List[DetectionBox]] = None


class Base64ImageRequest(BaseModel):
    """Request body for the Base64 OCR endpoint."""
    image: str = Field(
        ...,
        description="Base64-encoded image string (JPEG, PNG, or WebP). "
                    "Can optionally include the data URI prefix: "
                    "'data:image/jpeg;base64,...'"
    )
    include_raw: bool = Field(False, description="Include raw YOLO detections in response")

    @field_validator("image")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        # Strip data URI prefix if present: "data:image/jpeg;base64,<data>"
        if "," in v:
            v = v.split(",", 1)[1]
        # Validate it's valid base64
        try:
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("Invalid Base64 string.")
        return v


class PropertyItem(BaseModel):
    """A single name/value property in the Etisalat response format."""
    name: str
    value: Optional[str] = None


class ElsewedyOCRResult(BaseModel):
    """
    Client-facing response format.

    Each extracted field is an element in the ``properties`` list so that
    downstream consumers can iterate without knowing field names in advance.
    """
    description: str = "TEXT_DETECTION fetched data by ElsewedyOCR"
    properties: List[PropertyItem]


class CombinedIDRequest(BaseModel):
    """Request body for the combined front+back OCR endpoint."""
    front: str = Field(
        ...,
        description="Base64-encoded front of the ID card (JPEG/PNG/WebP). "
                    "Data URI prefix is accepted."
    )
    back: str = Field(
        ...,
        description="Base64-encoded back of the ID card (JPEG/PNG/WebP). "
                    "Data URI prefix is accepted."
    )
    include_raw: bool = Field(False, description="Include raw YOLO detections in response")

    @field_validator("front", "back")
    @classmethod
    def _strip_and_validate(cls, v: str) -> str:
        if "," in v:
            v = v.split(",", 1)[1]
        try:
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("Invalid Base64 string.")
        return v


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
