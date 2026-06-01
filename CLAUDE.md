# Arabic ID Card OCR — Project Guide for Claude

## What This Project Does

Extracts structured data from **Egyptian National ID cards** using a two-stage pipeline:

1. **YOLOv11** detects and localises each field on the ID card (bounding boxes)
2. **PaddleOCR** reads the Arabic/English text from each detected field crop

The result is a structured JSON with field names and their extracted text values.

---

## Architecture

```
Camera / Image
      │
      ▼
┌─────────────────────────┐
│   FieldDetector (YOLO)  │  YOLOv11n — detects 15 field regions
│   src/models/detector.py│  Model: runs/train/arabic_id_detector/weights/best.pt
└──────────┬──────────────┘
           │ bounding boxes (xyxy)
           ▼
┌─────────────────────────┐
│  enhance_for_ocr()      │  Upscale, deskew (±8° max), sharpen, CLAHE, denoise
│  src/data/preprocess.py │  field_type='numeric' adds morphological closing
└──────────┬──────────────┘
           │ enhanced RGB crop per field
           ▼
┌─────────────────────────┐
│  PaddleOCREngine        │  TextRecognition-only (arabic_PP-OCRv5_mobile_rec)
│  src/models/ocr_engine.py│  Skips internal detection — YOLO already localised fields
└──────────┬──────────────┘
           │ raw text + confidence per field
           ▼
┌─────────────────────────┐
│  postprocess_fields()   │  Per-field cleaning and validation
│  src/models/postprocess.py│
└──────────┬──────────────┘
           │ structured JSON
           ▼
      OCR Result
```

---

## Field Classes (15 total)

| Field | Type | Notes |
|---|---|---|
| `First_Name` | Arabic text | Name cleaning, rejects Latin garbage |
| `Last_Name` | Arabic text | Name cleaning |
| `HusbandName` | Arabic text | Optional field |
| `Job` | Arabic text | Occupation |
| `Gender` | Vocabulary | ذكر / أنثى — fuzzy matched |
| `Religion` | Vocabulary | مسلم / مسيحي etc — fuzzy matched |
| `Status` | Vocabulary | متزوج / أعزب etc — fuzzy matched |
| `Add1` | Arabic + digits | Address line 1 |
| `Add2` | Arabic + digits | Address line 2 |
| `ID` | 14 digits | Arabic-Indic → ASCII normalised |
| `Serial_Num` | Alphanumeric | Keeps letters (e.g. FI2430244) |
| `IssueDate` | DD/MM/YYYY | Date normalised from Arabic-Indic |
| `ExpDate` | DD/MM/YYYY | Date normalised |
| `Front` | Structural | Used to identify card side |
| `Back` | Structural | Used to identify card side |

---

## Key Files

```
OCR_Dataset/
├── configs/
│   ├── serving.yaml          ← Main config (OCR engine, model path, device)
│   └── training.yaml         ← YOLO training config
│
├── src/
│   ├── data/
│   │   └── preprocess.py     ← Image preprocessing (enhance_for_ocr, deskew, crop_field)
│   ├── models/
│   │   ├── detector.py       ← YOLOv11 wrapper (FieldDetector)
│   │   ├── ocr_engine.py     ← OCR engines: PaddleOCR, EasyOCR, Tesseract, TrOCR
│   │   ├── pipeline.py       ← ArabicIDOCRPipeline (wires detector + OCR + postprocess)
│   │   └── postprocess.py    ← Field-specific text cleaning and validation
│   └── serving/
│       ├── app.py            ← FastAPI REST API (/ocr, /ocr/batch, /health)
│       └── test_inference.py ← Camera/image test script with MLflow logging
│
├── scripts/
│   ├── generate_arabic_synthetic.py  ← Generates synthetic Arabic text images for OCR training
│   └── collect_ocr_crops.py          ← Crops fields using YOLO + auto-labels with PaddleOCR
│
├── notebooks/
│   └── 04_finetune_trocr_arabic.ipynb  ← Colab notebook to fine-tune TrOCR on Arabic
│
├── runs/train/arabic_id_detector/
│   └── weights/best.pt       ← Trained YOLO model (NOT in git — copy manually)
│
├── requirements-server.txt   ← Minimal deps for production server
└── CLAUDE.md                 ← This file
```

---

## Config: `configs/serving.yaml`

The most important config file. Controls which OCR engine is used:

```yaml
model:
  detector_path: ./runs/train/arabic_id_detector/weights/best.pt
  device: cpu
  ocr_engine: paddleocr    # paddleocr | easyocr | tesseract | trocr
  ocr_languages: ["ar", "en"]
  ocr_gpu: false
```

**Always use `paddleocr`** — it has the best Arabic accuracy (`arabic_PP-OCRv5_mobile_rec`).

---

## How to Run

### Camera test (interactive)
```bash
python3 src/serving/test_inference.py --camera
# SPACE = capture frame and run OCR
# Q = quit
```

### Single image test
```bash
python3 src/serving/test_inference.py --image path/to/id_card.jpg
```

### FastAPI server
```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
# POST /ocr          — single image
# POST /ocr/batch    — up to 10 images
# GET  /health       — health check
```

---

## Server Setup (fresh clone)

```bash
git clone https://github.com/OmarElGohary/arabic-id-ocr.git
cd arabic-id-ocr
git checkout azure-ml

python3 -m venv venv
source venv/bin/activate

pip install -r requirements-server.txt
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Copy YOLO weights (not in git — must be transferred manually)
mkdir -p runs/train/arabic_id_detector/weights/
cp /path/to/best.pt runs/train/arabic_id_detector/weights/best.pt

python3 src/serving/test_inference.py --camera
```

---

## YOLO Model

- **Architecture**: YOLOv11n (nano — lightweight, CPU-friendly)
- **Trained**: 100 epochs on Google Colab T4 GPU
- **mAP50**: 0.937 overall
- **Best classes**: Add1/Add2/Last_Name/Serial_Num (0.992–0.995)
- **Weakest classes**: Status (0.814), IssueDate (0.852)
- **Input size**: 640×640

### Per-class mAP50
| Class | mAP50 |
|---|---|
| Add1, Add2, Front, Last_Name, HusbandName | 0.995 |
| Serial_Num | 0.992 |
| First_Name | 0.993 |
| ID | 0.975 |
| Back | 0.970 |
| Gender | 0.892 |
| ExpDate | 0.869 |
| Religion | 0.867 |
| IssueDate | 0.852 |
| Job | 0.850 |
| Status | 0.814 |

---

## OCR Engine Details

### PaddleOCR (current — recommended)
- Uses `TextRecognition` only (no internal text detection)
- Model: `arabic_PP-OCRv5_mobile_rec`
- Input: BGR numpy array
- Init: `TextRecognition(model_name="arabic_PP-OCRv5_mobile_rec")`

### Why TextRecognition-only?
PaddleOCR's full pipeline runs its own text detector internally. When given a small pre-cropped field image (already detected by YOLO), the internal detector adds noise and slows things down. Using `TextRecognition` directly skips detection and runs the Arabic recognition model straight on the crop — faster and more accurate.

### Other engines (available but not recommended)
- `easyocr` — good general Arabic, but slower than PaddleOCR
- `tesseract` — lightest (100MB RAM), poor on ID card fonts without fine-tuning
- `trocr` — transformer-based, needs ~2GB RAM, not yet fine-tuned for Arabic

---

## Post-processing Rules

Each field has specific validation in `src/models/postprocess.py`:

| Field | Rule |
|---|---|
| `ID` | Strip non-digits, convert Arabic-Indic to ASCII, validate 13-14 digits |
| `Serial_Num` | Keep alphanumeric (letters + digits), normalize Arabic-Indic |
| `IssueDate` / `ExpDate` | Normalize to DD/MM/YYYY, validate ranges |
| `First_Name` / `Last_Name` / `HusbandName` / `Job` | Keep Arabic letters only, reject if < 2 chars or mostly Latin |
| `Gender` | Fuzzy match to: ذكر / أنثى |
| `Religion` | Fuzzy match to: مسلم / مسيحي / مسلمة / مسيحية / يهودي / أخرى |
| `Status` | Fuzzy match to: متزوج / أعزب / أرمل etc |
| `Add1` / `Add2` | Normalize numerals, keep Arabic + digits, light cleanup |

All fields strip invisible Unicode bidirectional marks that PaddleOCR injects.

---

## MLflow Tracking

Inference runs are logged automatically when using `test_inference.py`.

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Logged metrics per run:
- `fields_detected`, `fields_read`, `text_read_rate`
- `processing_time_s`
- Per-field read success flags (`read_First_Name`, `read_ID`, etc.)

---

## Fine-tuning OCR (Planned)

To improve OCR on blurry/poor-quality camera images:

1. Generate synthetic Arabic training data:
   ```bash
   python3 scripts/generate_arabic_synthetic.py --count 15000 --out data/arabic_synthetic
   cd data && zip -r arabic_synthetic.zip arabic_synthetic/
   ```

2. Upload `arabic_synthetic.zip` to Google Drive

3. Run `notebooks/04_finetune_trocr_arabic.ipynb` on Colab (T4 GPU, ~1-2 hours)
   - Architecture: `google/vit-base-patch16-224` encoder + `aubmindlab/bert-base-arabertv2` decoder
   - Augmentation: blur, noise, brightness, rotation, JPEG compression
   - Metric: CER (Character Error Rate) — lower is better

4. Download fine-tuned model, unzip to `models/trocr-arabic-id/`

5. Set `ocr_engine: trocr` in `configs/serving.yaml`

---

## Branch Structure

| Branch | Purpose |
|---|---|
| `main` | Local MPS (Apple Silicon) training config |
| `azure-ml` | Production branch — all improvements, CPU-compatible |

Always work on `azure-ml`.

---

## Known Limitations

- **Camera glare/reflections**: ID cards are shiny — diffuse lighting improves results significantly
- **ID number**: Arabic-Indic numerals read at ~20% accuracy (security watermark background interferes)
- **Dates**: ~13% read rate (same background issue)
- **Back side**: Not fully tested — YOLO detects `Back` class but back fields may need more training
- **TrOCR fine-tuning**: Not yet done — pending Colab training run
