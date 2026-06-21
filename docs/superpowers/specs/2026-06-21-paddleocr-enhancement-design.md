# PaddleOCR Enhancement — Tashkeel Recognition + Data Augmentation

**Date:** 2026-06-21
**Status:** Approved

## Goal

Improve PaddleOCR recognition accuracy on Egyptian ID card fields by:
1. Training the model to correctly recognize tashkeel (Arabic diacritics) when present in printed text
2. Expanding training vocabulary coverage (names, addresses, jobs, numeric/serial formats)
3. Confirming the existing data augmentation pipeline is sufficient

This is OCR *recognition* training — teaching the model to read diacritic glyphs already printed in an image. It is not automatic diacritization (adding diacritics to plain text), which is a separate NLP task and out of scope.

## Existing infrastructure (no need to rebuild)

The project already has a working PaddleOCR fine-tuning pipeline:

| File | Purpose |
|---|---|
| `scripts/generate_arabic_synthetic.py` | Synthetic Arabic text image generator with built-in augmentation |
| `scripts/collect_ocr_crops.py` | Auto-labels real ID card crops using YOLO + current PaddleOCR |
| `scripts/prepare_paddle_finetune.py` | Merges real + synthetic data into PaddleOCR training format (`train_label.txt`, `val_label.txt`, `arabic_dict.txt`) |
| `configs/paddle_rec_finetune.yml` | PaddleOCR SVTR_LCNet training config, matches `arabic_PP-OCRv5_mobile_rec` architecture |

Augmentation already implemented in both `generate_arabic_synthetic.py::augment()` and `prepare_paddle_finetune.py::augment_crop()`: Gaussian blur, motion blur, salt-and-pepper noise, perspective skew, rotation, brightness/contrast, Gaussian noise, JPEG compression artifacts — all independently randomized so they combine. **No changes needed to augmentation.**

## Part 1 — Data generation changes

### 1.1 Tashkeel via mishkal

- Add `mishkal` (pure-Python, rule/lexicon-based Arabic diacritizer) as a dependency. Chosen over `camel-tools` to avoid adding transformer/torch version pins that risk conflicting with this project's already-pinned CPU torch setup.
- Add `diacritize(text: str) -> str` helper in `generate_arabic_synthetic.py` that runs text through mishkal.
- In `sample_text()`, apply `diacritize()` to the sampled string with ~35% probability, across **all** Arabic text categories (names, addresses, jobs, gender/religion/status vocab) — not limited to specific fields. This produces a realistic mix of plain and diacritized training samples.
- No change needed to character dictionary building in `prepare_paddle_finetune.py::build_char_dict()` — diacritic codepoints (U+064B–U+0652) already fall inside the `0x0600–0x06FF` Arabic block it scans for, so they are picked up automatically once they appear in any label text.

### 1.2 Serial number format fix

- Current `random_serial()` generates digits only via `random_arabic_indic_number()`. Real Egyptian ID serials mix letters and digits (e.g. `FI2430244`, per `CLAUDE.md`'s field documentation for `Serial_Num`).
- Fix: generate 1–2 uppercase Latin letters + 6–8 digits to match the real format. This is a bug fix, not a new feature — the synthetic generator was never producing realistic serial samples.

### 1.3 Vocabulary expansion

- Roughly double `ARABIC_NAMES`, `ARABIC_ADDRESSES`, `ARABIC_JOBS` lists with more common Egyptian names, governorates/streets, and occupations — current lists have ~95 total entries combined, which limits diversity even with augmentation stacking.

### 1.4 Generation scale

- Bump default `--count` for `generate_arabic_synthetic.py` from 10,000 to 20,000, since the tashkeel variant roughly doubles effective vocabulary diversity (plain + diacritized versions of the same words).

## Part 2 — Training, evaluation, deployment

### 2.1 Training notebook

New `notebooks/05_finetune_paddleocr_arabic.ipynb`, parallel in structure to the existing `04_finetune_trocr_arabic.ipynb`, runs on Colab GPU (T4):

1. Clone the PaddleOCR repo, install dependencies including `mishkal`
2. Run `scripts/generate_arabic_synthetic.py --count 20000` (enhanced version from Part 1)
3. Run `scripts/collect_ocr_crops.py` against the real Egyptian ID dataset (refreshes real-crop labels using the current production model — script itself is unchanged, just re-run with current weights)
4. Run `scripts/prepare_paddle_finetune.py` to merge real + synthetic into PaddleOCR training format
5. Run `tools/train.py -c configs/paddle_rec_finetune.yml`, with `pretrained_model` filled in to point at the current `arabic_PP-OCRv5_mobile_rec` weights (config file itself needs no other changes)
6. Export the inference model, zip, download

### 2.2 Evaluation

Before promoting the new model:
- Compute Character Error Rate (CER) for both the current production model and the newly fine-tuned model, on the held-out validation split produced by `prepare_paddle_finetune.py`
- Report CER separately for the tashkeel-containing subset of the validation set, so the actual improvement on the targeted capability is visible, not just buried in an aggregate score

### 2.3 Deployment

- Update `PaddleOCREngine.__init__()` in `src/models/ocr_engine.py` to accept a custom model directory (`TextRecognition(model_dir=...)`) instead of only the hardcoded `model_name="arabic_PP-OCRv5_mobile_rec"` — allows pointing at the fine-tuned export without code changes elsewhere.
- Add the new model's path to `configs/serving.yaml`.
- Register the fine-tuned model in MLflow, reusing the existing pattern in `scripts/register_model.py`, before flipping production traffic to it — gives version history and rollback capability consistent with how the YOLO detector is already registered.

## Out of scope

- Automatic diacritization of plain (undiacritized) OCR output — a separate NLP task, not OCR recognition training
- camel-tools or other neural diacritization libraries — rejected due to dependency-conflict risk with this project's pinned CPU-only torch/numpy versions
- Changes to the existing augmentation functions — already comprehensive and cover the requested techniques (salt-and-pepper, blur, skew, and combinations)

## Success criteria

- Fine-tuned model's CER on the tashkeel subset of the validation set is measurably lower than the current production model's CER on the same subset
- No regression in CER on the non-tashkeel subset (existing capability preserved)
- Fine-tuned model registered in MLflow with version history before any production traffic switch
