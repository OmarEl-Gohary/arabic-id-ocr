"""Pydantic schemas for the OCR API request/response."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
    raw_detections: Optional[List[DetectionBox]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
