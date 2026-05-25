"""Integration tests for the FastAPI OCR endpoint."""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.serving import app as app_module


def make_jpeg_bytes(w=200, h=150) -> bytes:
    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_mock_pipeline():
    pipeline = MagicMock()
    pipeline.process_bytes.return_value = {
        "id_side": "front",
        "fields": {
            "First_Name": "أحمد",
            "Last_Name": "محمد",
            "ID": "29901011234567",
        },
        "metadata": {
            "num_fields_detected": 3,
            "processing_time_s": 0.5,
        },
        "raw_detections": [],
    }
    return pipeline


@pytest.fixture
def client():
    from src.serving.app import app
    with patch.object(app_module, "_pipeline", make_mock_pipeline()):
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_model_loaded(self, client):
        data = response = client.get("/health").json()
        assert "model_loaded" in data


class TestOCREndpoint:
    def test_ocr_valid_jpeg(self, client):
        response = client.post(
            "/ocr",
            files={"file": ("test.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "fields" in data
        assert "id_side" in data

    def test_ocr_unsupported_type_rejected(self, client):
        response = client.post(
            "/ocr",
            files={"file": ("test.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert response.status_code == 400

    def test_ocr_returns_arabic_text(self, client):
        response = client.post(
            "/ocr",
            files={"file": ("id.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        data = response.json()
        assert data["fields"].get("First_Name") == "أحمد"

    def test_ocr_metadata_present(self, client):
        response = client.post(
            "/ocr",
            files={"file": ("id.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        data = response.json()
        assert "metadata" in data
        assert "num_fields_detected" in data["metadata"]
