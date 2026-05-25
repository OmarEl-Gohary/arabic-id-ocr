# Arabic ID OCR — MLOps Project

End-to-end pipeline for extracting structured JSON data from Egyptian National ID cards.

## Architecture

**Two-stage inference:**
1. **YOLOv11** detects 15 field bounding boxes on the ID image
2. **EasyOCR** reads Arabic + English text from each cropped field
3. Results returned as a typed JSON dict

## Project Structure

```
src/
  data/           dataset.py · preprocess.py · validate.py
  models/         detector.py · ocr_engine.py · pipeline.py
  training/       train.py · evaluate.py
  serving/        app.py (FastAPI) · schemas.py
  monitoring/     metrics.py (Prometheus) · drift.py (Evidently)
pipelines/        training_pipeline.py · inference_pipeline.py  (Prefect)
configs/          model.yaml · training.yaml · serving.yaml
docker/           Dockerfile · docker-compose.yml
monitoring/       prometheus.yml · grafana/dashboards/
notebooks/        01_eda · 02_model_training · 03_inference_demo
tests/            test_data · test_models · test_api
```

## Classes (15)
`Add1 Add2 Back ExpDate First_Name Front Gender HusbandName ID IssueDate Job Last_Name Religion Serial_Num Status`

Front-side fields: First_Name, Last_Name, ID, Gender, Religion, HusbandName, Job
Back-side fields: Add1, Add2, Serial_Num, IssueDate, ExpDate, Status

## Dataset
- Source: Roboflow — Egyptian Person ID v1 (3520 images, YOLOv11 format)
- Path: `Egyptain-Person-ID-1/` (train/valid/test splits)
- Augmentations already applied at export (flips, rotations, blur, brightness)

## MLOps Stack
| Concern | Tool |
|---|---|
| Data versioning | DVC |
| Experiment tracking | MLflow |
| Pipeline orchestration | Prefect |
| Model serving | FastAPI + Uvicorn |
| Containerization | Docker Compose |
| Metrics | Prometheus + Grafana |
| Drift detection | Evidently AI |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Validate data
make validate-data

# Train (GPU recommended)
make train

# Evaluate
make evaluate

# Serve API
make serve          # http://localhost:8000/docs

# Full stack
make docker-up      # API + MLflow + Prometheus + Grafana + Prefect
```

## API Usage

```bash
# Single image
curl -X POST http://localhost:8000/ocr \
  -F "file=@id_card.jpg" | python -m json.tool

# Response shape
{
  "id_side": "front",
  "fields": {
    "First_Name": "أحمد",
    "Last_Name": "محمد",
    "ID": "29901011234567",
    "Gender": "ذكر",
    "Religion": "مسلم"
  },
  "metadata": {
    "num_fields_detected": 8,
    "processing_time_s": 1.2
  }
}
```

## Training Config
Edit `configs/training.yaml` to change epochs, batch size, learning rate.
Default: `yolo11n` (nano, fast) — swap to `yolo11s` or `yolo11m` for higher accuracy.

## Running Tests

```bash
make test           # all tests
make test-cov       # with coverage report
```

## DVC Pipeline

```bash
dvc repro           # runs validate → train → evaluate in order
dvc dag             # view pipeline DAG
```
