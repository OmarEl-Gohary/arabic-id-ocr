# PaddleOCR Tashkeel Recognition Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing PaddleOCR fine-tuning data pipeline to generate tashkeel-aware synthetic training data (via `mishkal`), fix a latent serial-number format bug, add shadow simulation to augmentation, expand training vocabulary, and wire a custom fine-tuned model path end-to-end from training notebook through to deployment + MLflow registration.

**Architecture:** No new subsystems. This extends four existing files (`scripts/generate_arabic_synthetic.py`, `scripts/prepare_paddle_finetune.py`, `src/models/ocr_engine.py`, `scripts/register_model.py`) and `configs/serving.yaml`, adds one new small module (`src/evaluation/cer.py` + thin CLI wrapper `scripts/evaluate_cer.py`), and adds one new Colab notebook (`notebooks/05_finetune_paddleocr_arabic.ipynb`) that orchestrates the already-existing scripts plus PaddleOCR's own `tools/train.py`.

**Tech Stack:** Python 3.11, PaddleOCR 3.x (`TextRecognition`), PaddlePaddle (training on Colab GPU), `mishkal` (Arabic diacritization), PIL/OpenCV (image augmentation), pytest (testing), MLflow (model registry).

**Spec:** `docs/superpowers/specs/2026-06-21-paddleocr-enhancement-design.md`

---

## Task 1: Fix `random_serial()` to generate realistic alphanumeric format

Real Egyptian ID serials mix letters and digits (e.g. `FI2430244`), but the current generator only produces digits. This is a latent bug independent of the tashkeel work.

**Files:**
- Modify: `scripts/generate_arabic_synthetic.py:78-79`
- Test: `tests/test_synthetic_generation.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_synthetic_generation.py`:

```python
"""Tests for synthetic Arabic OCR training data generation."""

import random
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import generate_arabic_synthetic as gen


class TestRandomSerial:
    def test_random_serial_matches_real_format(self):
        for _ in range(50):
            s = gen.random_serial()
            assert re.match(r"^[A-Z]{1,2}[0-9]{6,8}$", s), f"Serial {s!r} has wrong format"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthetic_generation.py::TestRandomSerial -v`
Expected: FAIL — current `random_serial()` returns Arabic-Indic digits only, no letters, so the `[A-Z]{1,2}` part of the regex never matches.

- [ ] **Step 3: Write minimal implementation**

In `scripts/generate_arabic_synthetic.py`, replace lines 78-79:

```python
def random_serial() -> str:
    return random_arabic_indic_number(random.randint(7, 9))
```

with:

```python
LATIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def random_serial() -> str:
    """Egyptian ID serial format: 1-2 uppercase letters + 6-8 digits, e.g. FI2430244."""
    n_letters = random.randint(1, 2)
    n_digits = random.randint(6, 8)
    letters = "".join(random.choice(LATIN_LETTERS) for _ in range(n_letters))
    digits = "".join(random.choice("0123456789") for _ in range(n_digits))
    return letters + digits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthetic_generation.py::TestRandomSerial -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_arabic_synthetic.py tests/test_synthetic_generation.py
git commit -m "fix: generate alphanumeric serial numbers matching real ID format"
```

---

## Task 2: Add `diacritize()` using mishkal

**Files:**
- Modify: `scripts/generate_arabic_synthetic.py` (add near top, after imports)
- Modify: `requirements.txt` (add `mishkal` dependency)
- Test: `tests/test_synthetic_generation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_synthetic_generation.py` (the `unittest.mock` import is already present from Task 1):

```python
class TestDiacritize:
    def test_diacritize_calls_vocalizer(self):
        fake_vocalizer = MagicMock()
        fake_vocalizer.tashkeel.return_value = "مُحَمَّد"
        with patch.object(gen, "_get_vocalizer", return_value=fake_vocalizer):
            result = gen.diacritize("محمد")
        assert result == "مُحَمَّد"
        fake_vocalizer.tashkeel.assert_called_once_with("محمد")

    def test_diacritize_falls_back_to_original_on_error(self):
        with patch.object(gen, "_get_vocalizer", side_effect=RuntimeError("boom")):
            result = gen.diacritize("محمد")
        assert result == "محمد"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthetic_generation.py::TestDiacritize -v`
Expected: FAIL with `AttributeError: module 'generate_arabic_synthetic' has no attribute 'diacritize'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/generate_arabic_synthetic.py`, add after the `ARABIC_INDIC` constant (after line 60):

```python
# ── Tashkeel (diacritics) via mishkal ──────────────────────────────────────────

_TASHKEEL_VOCALIZER = None


def _get_vocalizer():
    """Lazily load and cache the mishkal vocalizer (loading its lexicon is slow)."""
    global _TASHKEEL_VOCALIZER
    if _TASHKEEL_VOCALIZER is None:
        from mishkal import tashkeel
        _TASHKEEL_VOCALIZER = tashkeel.TashkeelClass()
    return _TASHKEEL_VOCALIZER


def diacritize(text: str) -> str:
    """
    Add tashkeel (diacritics) to Arabic text using mishkal.

    Falls back to the original text if mishkal is unavailable or fails to
    vocalize a particular string — never raises.
    """
    try:
        vocalizer = _get_vocalizer()
        return vocalizer.tashkeel(text)
    except Exception:
        return text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthetic_generation.py::TestDiacritize -v`
Expected: PASS

- [ ] **Step 5: Add mishkal to requirements.txt**

In `requirements.txt`, add a new line (alphabetically near the "m" entries, e.g. after `mistune==3.2.1` and before `mlflow==3.12.0`):

```
mishkal==0.4.3
```

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_arabic_synthetic.py requirements.txt tests/test_synthetic_generation.py
git commit -m "feat: add mishkal-based tashkeel diacritization for synthetic data"
```

---

## Task 3: Wire `diacritize()` into `sample_text()` at 30% probability

Without-tashkeel must be the majority class (~70%) since real Egyptian ID cards are printed without diacritics in the vast majority of cases.

**Files:**
- Modify: `scripts/generate_arabic_synthetic.py:82-114`
- Test: `tests/test_synthetic_generation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_synthetic_generation.py` (the `random` import is already present from Task 1):

```python
class TestSampleTextTashkeelRatio:
    def test_sample_text_applies_tashkeel_around_30_percent(self):
        fake_vocalizer = MagicMock()
        fake_vocalizer.tashkeel.side_effect = lambda t: t + "َ"  # append fatha

        random.seed(42)
        with patch.object(gen, "_get_vocalizer", return_value=fake_vocalizer):
            samples = [gen.sample_text() for _ in range(2000)]

        diacritized = sum(1 for s in samples if "َ" in s)
        ratio = diacritized / len(samples)
        assert 0.22 <= ratio <= 0.38, f"Tashkeel ratio {ratio:.2f} outside expected ~30% band"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthetic_generation.py::TestSampleTextTashkeelRatio -v`
Expected: FAIL — `sample_text()` never calls `diacritize()`, so `ratio` is 0.0.

- [ ] **Step 3: Write minimal implementation**

In `scripts/generate_arabic_synthetic.py`, modify `sample_text()` (lines 82-114):

```python
def sample_text() -> str:
    """Return a random Arabic text sample covering all field types."""
    category = random.choices(
        ["name", "address", "job", "gender", "religion", "status",
         "id", "date", "serial", "multi_word"],
        weights=[25, 20, 10, 5, 5, 5, 10, 10, 5, 5],
    )[0]

    if category == "name":
        n = random.randint(1, 3)
        text = " ".join(random.choice(ARABIC_NAMES) for _ in range(n))
    elif category == "address":
        parts = random.sample(ARABIC_ADDRESSES, random.randint(1, 2))
        prefix = random_arabic_indic_number(random.randint(1, 2)) + " " if random.random() < 0.4 else ""
        text = prefix + " ".join(parts)
    elif category == "job":
        text = random.choice(ARABIC_JOBS)
    elif category == "gender":
        text = random.choice(GENDER_VALUES)
    elif category == "religion":
        text = random.choice(RELIGION_VALUES)
    elif category == "status":
        text = random.choice(STATUS_VALUES)
    elif category == "id":
        return random_id()
    elif category == "date":
        return random_date()
    elif category == "serial":
        return random_serial()
    else:
        text = " ".join(random.choice(ARABIC_NAMES) for _ in range(random.randint(2, 4)))

    # Tashkeel: ~30% of Arabic-text samples are diacritized (minority class —
    # real ID cards are printed without diacritics in most cases).
    if random.random() < 0.30:
        text = diacritize(text)
    return text
```

Note: numeric categories (`id`, `date`, `serial`) return early before the tashkeel branch since they contain no Arabic letters to diacritize.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthetic_generation.py::TestSampleTextTashkeelRatio -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_arabic_synthetic.py tests/test_synthetic_generation.py
git commit -m "feat: apply tashkeel to ~30% of generated Arabic text samples"
```

---

## Task 4: Add `add_shadow()` augmentation to `generate_arabic_synthetic.py`

**Files:**
- Modify: `scripts/generate_arabic_synthetic.py:184-258` (the `augment()` function)
- Test: `tests/test_synthetic_generation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_synthetic_generation.py`:

```python
class TestAddShadow:
    def test_add_shadow_preserves_size_and_mode(self):
        img = Image.new("RGB", (150, 80), (200, 200, 200))
        out = gen.add_shadow(img)
        assert out.size == (150, 80)
        assert out.mode == "RGB"

    def test_add_shadow_darkens_some_pixels(self):
        img = Image.new("RGB", (200, 100), (255, 255, 255))
        random.seed(7)
        out = gen.add_shadow(img)
        arr_before = np.array(img)
        arr_after = np.array(out)
        assert not np.array_equal(arr_before, arr_after)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthetic_generation.py::TestAddShadow -v`
Expected: FAIL with `AttributeError: module 'generate_arabic_synthetic' has no attribute 'add_shadow'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/generate_arabic_synthetic.py`, add a new function just before `augment()` (before line 184):

```python
def add_shadow(img: Image.Image) -> Image.Image:
    """Overlay a soft-edged dark gradient blob simulating a partial shadow
    from a phone, finger, or uneven lighting during capture."""
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    cx, cy = random.randint(0, w), random.randint(0, h)
    rx = random.randint(max(1, w // 3), w)
    ry = random.randint(max(1, h // 3), h)
    draw.ellipse(
        [cx - rx, cy - ry, cx + rx, cy + ry],
        fill=random.randint(80, 160),
    )
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(w, h) // 6 + 1))

    dark = Image.new("RGB", (w, h), (0, 0, 0))
    return Image.composite(dark, img, mask).convert("RGB")
```

Then in `augment()`, add the shadow step right after the rotation block (after the line `img = img.rotate(angle, fillcolor=(255, 255, 255), expand=False)` and its closing of that `if`, currently around line 248), and before the JPEG compression block:

```python
    # Shadow (partial shadow from phone/finger/object during capture)
    if random.random() < 0.3:
        img = add_shadow(img)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthetic_generation.py::TestAddShadow -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_arabic_synthetic.py tests/test_synthetic_generation.py
git commit -m "feat: add shadow simulation to synthetic data augmentation"
```

---

## Task 5: Mirror `add_shadow()` into `prepare_paddle_finetune.py::augment_crop()`

The two scripts maintain independent augmentation functions (existing pattern — `augment()` and `augment_crop()` already duplicate blur/skew/noise/etc. rather than sharing a module). Follow that pattern for consistency.

**Files:**
- Modify: `scripts/prepare_paddle_finetune.py:39-99` (the `augment_crop()` function)
- Test: `tests/test_synthetic_generation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_synthetic_generation.py`:

```python
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import prepare_paddle_finetune as prep


class TestAugmentCropShadow:
    def test_augment_crop_has_add_shadow_step(self):
        # add_shadow must exist and be usable standalone in this module too
        img = Image.new("RGB", (120, 60), (180, 180, 180))
        out = prep.add_shadow(img)
        assert out.size == (120, 60)
        assert out.mode == "RGB"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthetic_generation.py::TestAugmentCropShadow -v`
Expected: FAIL with `AttributeError: module 'prepare_paddle_finetune' has no attribute 'add_shadow'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/prepare_paddle_finetune.py`, add the same `add_shadow()` function used in Task 4, placed just before `augment_crop()` (before line 39):

```python
def add_shadow(img: Image.Image) -> Image.Image:
    """Overlay a soft-edged dark gradient blob simulating a partial shadow
    from a phone, finger, or uneven lighting during capture."""
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    cx, cy = random.randint(0, w), random.randint(0, h)
    rx = random.randint(max(1, w // 3), w)
    ry = random.randint(max(1, h // 3), h)
    draw.ellipse(
        [cx - rx, cy - ry, cx + rx, cy + ry],
        fill=random.randint(80, 160),
    )
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(w, h) // 6 + 1))

    dark = Image.new("RGB", (w, h), (0, 0, 0))
    return Image.composite(dark, img, mask).convert("RGB")
```

Note: `ImageDraw` is not yet imported in this file — add it to the existing import line. Change:

```python
from PIL import Image, ImageEnhance, ImageFilter
```

to:

```python
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
```

Then in `augment_crop()`, add the shadow step right after the rotation block (after `img = img.rotate(random.uniform(-6, 6), fillcolor=(255, 255, 255), expand=False)`, currently around line 90) and before the JPEG compression block:

```python
    # Shadow (partial shadow from phone/finger/object during capture)
    if random.random() < 0.3:
        img = add_shadow(img)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthetic_generation.py::TestAugmentCropShadow -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_paddle_finetune.py tests/test_synthetic_generation.py
git commit -m "feat: add shadow simulation to real-crop augmentation pipeline"
```

---

## Task 6: Expand vocabulary lists

**Files:**
- Modify: `scripts/generate_arabic_synthetic.py:28-51`
- Test: `tests/test_synthetic_generation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_synthetic_generation.py`:

```python
class TestVocabularyExpansion:
    def test_names_list_expanded(self):
        assert len(gen.ARABIC_NAMES) >= 80

    def test_addresses_list_expanded(self):
        assert len(gen.ARABIC_ADDRESSES) >= 45

    def test_jobs_list_expanded(self):
        assert len(gen.ARABIC_JOBS) >= 35

    def test_no_duplicate_names(self):
        assert len(gen.ARABIC_NAMES) == len(set(gen.ARABIC_NAMES))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthetic_generation.py::TestVocabularyExpansion -v`
Expected: FAIL — current lists have 44 names, 31 addresses, 20 jobs.

- [ ] **Step 3: Write minimal implementation**

In `scripts/generate_arabic_synthetic.py`, replace lines 28-51:

```python
ARABIC_NAMES = [
    "محمد", "أحمد", "علي", "عمر", "إبراهيم", "خالد", "يوسف", "عبدالله",
    "مصطفى", "حسن", "حسين", "طارق", "كريم", "سامي", "وليد", "هاني",
    "فاطمة", "مريم", "نور", "سارة", "هند", "دينا", "رانيا", "أميرة",
    "ياسمين", "منى", "سمر", "غادة", "نادية", "هالة", "إيمان", "شيماء",
    "عبدالرحمن", "عبدالعزيز", "عبدالكريم", "عبدالحميد", "شحاتة", "سيد",
    "البدوي", "الشيخ", "الحسيني", "الأنصاري", "الشافعي", "الزيات",
]

ARABIC_ADDRESSES = [
    "شارع التحرير", "ميدان رمسيس", "شارع النيل", "شارع السلام",
    "شارع العروبة", "ميدان الجيزة", "شارع البحر", "حي العجوزة",
    "شارع الجمهورية", "شارع الهرم", "حي الدقي", "حي المعادي",
    "شارع قصر النيل", "حي الزيتون", "حي شبرا", "حي حلوان",
    "الاسكندرية", "القاهرة", "الجيزة", "الإسماعيلية", "السويس",
    "المنصورة", "طنطا", "أسيوط", "المنيا", "سوهاج", "أسوان",
    "قنا", "الفيوم", "بني سويف", "دمياط", "كفر الشيخ",
]

ARABIC_JOBS = [
    "مهندس", "طبيب", "محامٍ", "معلم", "موظف", "تاجر", "محاسب",
    "مدير", "فني", "كاتب", "صيدلاني", "ممرض", "مقاول", "سائق",
    "عامل", "متقاعد", "ربة منزل", "طالب", "أستاذ جامعي", "باحث",
]
```

with:

```python
ARABIC_NAMES = [
    "محمد", "أحمد", "علي", "عمر", "إبراهيم", "خالد", "يوسف", "عبدالله",
    "مصطفى", "حسن", "حسين", "طارق", "كريم", "سامي", "وليد", "هاني",
    "فاطمة", "مريم", "نور", "سارة", "هند", "دينا", "رانيا", "أميرة",
    "ياسمين", "منى", "سمر", "غادة", "نادية", "هالة", "إيمان", "شيماء",
    "عبدالرحمن", "عبدالعزيز", "عبدالكريم", "عبدالحميد", "شحاتة", "سيد",
    "البدوي", "الشيخ", "الحسيني", "الأنصاري", "الشافعي", "الزيات",
    "محمود", "إسماعيل", "عبدالناصر", "جمال", "سعيد", "فؤاد", "أنور",
    "رفعت", "مجدي", "عماد", "شريف", "أيمن", "هشام", "رضا", "نبيل",
    "فتحي", "لطفي", "زكريا", "إلياس", "داود",
    "زينب", "خديجة", "آمنة", "ليلى", "نهى", "وفاء", "عزة", "سهام",
    "نجلاء", "نيفين", "جيهان", "مها", "رشا", "دعاء", "إسراء", "مروة",
    "أسماء", "بسمة", "نهال", "سلمى",
    "المصري", "السيد", "عبدالوهاب", "الجندي", "البربري", "الخولي",
    "العربي", "درويش", "قنديل", "زهران",
]

ARABIC_ADDRESSES = [
    "شارع التحرير", "ميدان رمسيس", "شارع النيل", "شارع السلام",
    "شارع العروبة", "ميدان الجيزة", "شارع البحر", "حي العجوزة",
    "شارع الجمهورية", "شارع الهرم", "حي الدقي", "حي المعادي",
    "شارع قصر النيل", "حي الزيتون", "حي شبرا", "حي حلوان",
    "الاسكندرية", "القاهرة", "الجيزة", "الإسماعيلية", "السويس",
    "المنصورة", "طنطا", "أسيوط", "المنيا", "سوهاج", "أسوان",
    "قنا", "الفيوم", "بني سويف", "دمياط", "كفر الشيخ",
    "شارع الستين", "شارع مصر حلوان الزراعي", "ميدان الساعة", "شارع فيصل",
    "حي مدينة نصر", "حي المهندسين", "حي الزمالك", "شارع الكورنيش",
    "حي العباسية", "حي روض الفرج", "الزقازيق", "دمنهور", "بنها",
    "المحلة الكبرى", "إدفو", "مرسى مطروح", "رأس البر", "شرم الشيخ",
    "الغردقة", "العريش",
]

ARABIC_JOBS = [
    "مهندس", "طبيب", "محامٍ", "معلم", "موظف", "تاجر", "محاسب",
    "مدير", "فني", "كاتب", "صيدلاني", "ممرض", "مقاول", "سائق",
    "عامل", "متقاعد", "ربة منزل", "طالب", "أستاذ جامعي", "باحث",
    "مبرمج", "مهندس معماري", "مصمم", "مترجم", "صحفي", "إعلامي",
    "مدرس", "مهندس كهرباء", "فني صيانة", "عامل بناء", "بائع",
    "حداد", "نجار", "سباك", "كهربائي", "طيار", "ربان",
    "ضابط شرطة", "جندي", "موظف حكومي",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthetic_generation.py::TestVocabularyExpansion -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_arabic_synthetic.py tests/test_synthetic_generation.py
git commit -m "feat: expand names/addresses/jobs vocabulary for synthetic data"
```

---

## Task 7: Bump default synthetic data generation count

**Files:**
- Modify: `scripts/generate_arabic_synthetic.py:312-313`

- [ ] **Step 1: Modify the argparse default**

In `scripts/generate_arabic_synthetic.py`, change:

```python
    p.add_argument("--count",      type=int, default=10000,
                   help="Number of synthetic images to generate")
```

to:

```python
    p.add_argument("--count",      type=int, default=20000,
                   help="Number of synthetic images to generate")
```

- [ ] **Step 2: Verify with a quick manual check**

Run: `python scripts/generate_arabic_synthetic.py --count 5 --out /tmp/syn_test`
Expected: Generates 5 images without error, prints "Found N font(s)" and completes — confirms the script still runs after all Task 1-6 changes.

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_arabic_synthetic.py
git commit -m "feat: bump default synthetic generation count to 20000"
```

---

## Task 8: Add CER evaluation module

**Files:**
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/cer.py`
- Test: `tests/test_evaluation.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluation.py`:

```python
"""Tests for OCR evaluation metrics (Character Error Rate)."""

from src.evaluation.cer import char_error_rate, has_tashkeel


class TestCharErrorRate:
    def test_identical_strings_zero_cer(self):
        assert char_error_rate("محمد", "محمد") == 0.0

    def test_empty_target_nonempty_pred_is_full_error(self):
        assert char_error_rate("محمد", "") == 1.0

    def test_empty_target_empty_pred_is_zero(self):
        assert char_error_rate("", "") == 0.0

    def test_one_substitution(self):
        # "محمض" vs "محمد" -> 1 edit out of 4 target chars
        cer = char_error_rate("محمض", "محمد")
        assert abs(cer - 0.25) < 1e-9

    def test_completely_wrong_prediction(self):
        cer = char_error_rate("xyz", "محمد")
        assert cer == 1.0


class TestHasTashkeel:
    def test_true_for_diacritized_text(self):
        assert has_tashkeel("مُحَمَّد") is True

    def test_false_for_plain_text(self):
        assert has_tashkeel("محمد") is False

    def test_false_for_empty_string(self):
        assert has_tashkeel("") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.evaluation'`

- [ ] **Step 3: Write minimal implementation**

Create `src/evaluation/__init__.py` (empty file, matches the existing pattern in `src/data/__init__.py` and `src/models/__init__.py`).

Create `src/evaluation/cer.py`:

```python
"""Character Error Rate (CER) computation for OCR evaluation."""

TASHKEEL_RANGE = range(0x064B, 0x0653)  # Arabic diacritic codepoints


def has_tashkeel(text: str) -> bool:
    """Return True if any character in text is an Arabic diacritic mark."""
    return any(ord(c) in TASHKEEL_RANGE for c in text)


def char_error_rate(pred: str, target: str) -> float:
    """
    Character Error Rate: Levenshtein edit distance divided by target length.

    Returns 0.0 for an empty target with an empty prediction, 1.0 for an
    empty target with a non-empty prediction.
    """
    if not target:
        return 0.0 if not pred else 1.0

    m, n = len(pred), len(target)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            if pred[i - 1] == target[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[n] / n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/__init__.py src/evaluation/cer.py tests/test_evaluation.py
git commit -m "feat: add CER evaluation module with tashkeel-subset detection"
```

---

## Task 9: Add `model_dir` support to `PaddleOCREngine`

**Files:**
- Modify: `src/models/ocr_engine.py:198-205`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_models.py`:

```python
class TestPaddleOCREngineModelDir:
    @patch("paddleocr.TextRecognition")
    def test_default_uses_builtin_model_name(self, mock_text_recognition):
        from src.models.ocr_engine import PaddleOCREngine
        PaddleOCREngine(lang="ar")
        mock_text_recognition.assert_called_once_with(model_name="arabic_PP-OCRv5_mobile_rec")

    @patch("paddleocr.TextRecognition")
    def test_custom_model_dir_used_when_provided(self, mock_text_recognition):
        from src.models.ocr_engine import PaddleOCREngine
        PaddleOCREngine(lang="ar", model_dir="/path/to/finetuned")
        mock_text_recognition.assert_called_once_with(model_dir="/path/to/finetuned")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py::TestPaddleOCREngineModelDir -v`
Expected: FAIL — `test_custom_model_dir_used_when_provided` fails with `TypeError: __init__() got an unexpected keyword argument 'model_dir'`

- [ ] **Step 3: Write minimal implementation**

In `src/models/ocr_engine.py`, replace lines 198-205:

```python
    def __init__(self, lang: str = "ar", use_gpu: bool = False):
        try:
            from paddleocr import TextRecognition
        except ImportError:
            raise ImportError("pip install paddlepaddle paddleocr>=3.0")

        self.rec  = TextRecognition(model_name="arabic_PP-OCRv5_mobile_rec")
        self.lang = lang
```

with:

```python
    def __init__(self, lang: str = "ar", use_gpu: bool = False, model_dir: Optional[str] = None):
        try:
            from paddleocr import TextRecognition
        except ImportError:
            raise ImportError("pip install paddlepaddle paddleocr>=3.0")

        if model_dir:
            self.rec = TextRecognition(model_dir=model_dir)
        else:
            self.rec = TextRecognition(model_name="arabic_PP-OCRv5_mobile_rec")
        self.lang = lang
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py::TestPaddleOCREngineModelDir -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/ocr_engine.py tests/test_models.py
git commit -m "feat: allow PaddleOCREngine to load a custom fine-tuned model_dir"
```

---

## Task 10: Wire `paddleocr_model_dir` through serving config

**Files:**
- Modify: `configs/serving.yaml`
- Modify: `src/serving/app.py` (the `load_pipeline()` function)
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api.py`:

```python
class TestLoadPipelinePaddleOCRModelDir:
    def test_load_pipeline_passes_model_dir_when_configured(self, tmp_path):
        cfg_path = tmp_path / "serving.yaml"
        cfg_path.write_text(
            "model:\n"
            "  detector_path: ./runs/train/arabic_id_detector/weights/best.pt\n"
            "  device: cpu\n"
            "  ocr_engine: paddleocr\n"
            "  ocr_languages: [\"ar\", \"en\"]\n"
            "  ocr_gpu: false\n"
            "  paddleocr_model_dir: /fake/finetuned/model\n",
            encoding="utf-8",
        )

        with patch.object(app_module, "SERVING_CONFIG", str(cfg_path)), \
             patch.object(app_module, "FieldDetector"), \
             patch.object(app_module, "create_ocr_engine") as mock_create_engine, \
             patch.object(app_module, "ArabicIDOCRPipeline"):
            app_module.load_pipeline()

        _, kwargs = mock_create_engine.call_args
        assert kwargs.get("model_dir") == "/fake/finetuned/model"

    def test_load_pipeline_omits_model_dir_when_not_configured(self, tmp_path):
        cfg_path = tmp_path / "serving.yaml"
        cfg_path.write_text(
            "model:\n"
            "  detector_path: ./runs/train/arabic_id_detector/weights/best.pt\n"
            "  device: cpu\n"
            "  ocr_engine: paddleocr\n"
            "  ocr_languages: [\"ar\", \"en\"]\n"
            "  ocr_gpu: false\n",
            encoding="utf-8",
        )

        with patch.object(app_module, "SERVING_CONFIG", str(cfg_path)), \
             patch.object(app_module, "FieldDetector"), \
             patch.object(app_module, "create_ocr_engine") as mock_create_engine, \
             patch.object(app_module, "ArabicIDOCRPipeline"):
            app_module.load_pipeline()

        _, kwargs = mock_create_engine.call_args
        assert "model_dir" not in kwargs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py::TestLoadPipelinePaddleOCRModelDir -v`
Expected: FAIL on `test_load_pipeline_passes_model_dir_when_configured` — `model_dir` is never in `kwargs` since `load_pipeline()` doesn't read `paddleocr_model_dir` yet.

- [ ] **Step 3: Write minimal implementation**

In `src/serving/app.py`, modify the `load_pipeline()` function's paddleocr branch:

```python
    elif ocr_engine == "paddleocr":
        engine_kwargs = {"lang": "ar", "use_gpu": ocr_gpu}
```

to:

```python
    elif ocr_engine == "paddleocr":
        engine_kwargs = {"lang": "ar", "use_gpu": ocr_gpu}
        paddleocr_model_dir = model_cfg.get("paddleocr_model_dir")
        if paddleocr_model_dir:
            engine_kwargs["model_dir"] = paddleocr_model_dir
```

In `configs/serving.yaml`, add a line after `ocr_gpu: false`:

```yaml
  ocr_gpu: false
  paddleocr_model_dir: null   # set to a local path to use a fine-tuned model instead of the built-in arabic_PP-OCRv5_mobile_rec
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py::TestLoadPipelinePaddleOCRModelDir -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/serving/app.py configs/serving.yaml tests/test_api.py
git commit -m "feat: wire paddleocr_model_dir config option through to serving pipeline"
```

---

## Task 11: Create `scripts/evaluate_cer.py` CLI wrapper

Thin CLI glue around the tested `src/evaluation/cer.py` functions and the existing `PaddleOCREngine`. No new unit tests needed here — matches the existing convention that other CLI scripts (`collect_ocr_crops.py`, `prepare_paddle_finetune.py`) are untested glue code calling already-tested logic.

**Files:**
- Create: `scripts/evaluate_cer.py`

- [ ] **Step 1: Create the script**

```python
"""
Compute Character Error Rate (CER) for a PaddleOCR recognition model
against a labeled validation set, with separate reporting for the
tashkeel-containing subset.

Usage:
    python scripts/evaluate_cer.py \
        --val-label data/paddle_finetune/val_label.txt \
        --val-dir   data/paddle_finetune/val \
        --model-name arabic_PP-OCRv5_mobile_rec

    python scripts/evaluate_cer.py \
        --val-label data/paddle_finetune/val_label.txt \
        --val-dir   data/paddle_finetune/val \
        --model-dir output/arabic_id_rec/inference
"""

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.cer import char_error_rate, has_tashkeel
from src.models.ocr_engine import PaddleOCREngine


def load_label_file(label_path: Path) -> list:
    rows = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            img_rel, text = line.split("\t", 1)
            rows.append((img_rel, text))
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val-label", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--model-name", default=None)
    p.add_argument("--model-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.model_name and not args.model_dir:
        sys.exit("Provide either --model-name or --model-dir")

    engine = PaddleOCREngine(lang="ar", model_dir=args.model_dir) if args.model_dir \
        else PaddleOCREngine(lang="ar")

    rows = load_label_file(Path(args.val_label))
    val_dir = Path(args.val_dir)

    all_cers, tashkeel_cers, plain_cers = [], [], []

    for img_rel, target in rows:
        img_path = val_dir / img_rel
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred, _ = engine.read_text_with_confidence(img_rgb)

        cer = char_error_rate(pred, target)
        all_cers.append(cer)
        (tashkeel_cers if has_tashkeel(target) else plain_cers).append(cer)

    def avg(values):
        return sum(values) / len(values) if values else float("nan")

    print(f"Samples evaluated   : {len(all_cers)}")
    print(f"Overall CER         : {avg(all_cers):.4f}")
    print(f"Tashkeel subset CER : {avg(tashkeel_cers):.4f}  (n={len(tashkeel_cers)})")
    print(f"Plain subset CER    : {avg(plain_cers):.4f}  (n={len(plain_cers)})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs (smoke test with --help)**

Run: `python scripts/evaluate_cer.py --help`
Expected: Prints usage/argument help without error — confirms imports resolve.

- [ ] **Step 3: Commit**

```bash
git add scripts/evaluate_cer.py
git commit -m "feat: add CER evaluation CLI script comparing old vs fine-tuned model"
```

---

## Task 12: Extend `scripts/register_model.py` to register the fine-tuned PaddleOCR model

**Files:**
- Modify: `scripts/register_model.py` (existing file, registers YOLO only today)

- [ ] **Step 1: Add the PaddleOCR registration function and CLI flag**

In `scripts/register_model.py`, add near the top (after `YOLO_WEIGHTS_PATH`):

```python
PADDLEOCR_MODEL_NAME = "arabic-paddleocr-rec"
```

Add a new function after `register_yolo()`:

```python
def register_paddleocr(client: MlflowClient, tracking_uri: str, model_dir: str) -> str:
    """Log a fine-tuned PaddleOCR recognition export and register it in Staging
    (not auto-promoted to Production — promote manually after CER review)."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"PaddleOCR model dir not found at {model_path}")

    with mlflow.start_run(run_name="paddleocr-rec-register") as run:
        mlflow.log_param("base_model", "arabic_PP-OCRv5_mobile_rec")
        mlflow.log_param("fine_tuned", True)
        mlflow.log_param("tashkeel_enhanced", True)
        mlflow.log_artifacts(str(model_path), artifact_path="inference")
        run_id = run.info.run_id

    try:
        client.create_registered_model(
            name=PADDLEOCR_MODEL_NAME,
            description="Fine-tuned arabic_PP-OCRv5_mobile_rec — tashkeel-aware recognition",
        )
        print(f"Created registered model: {PADDLEOCR_MODEL_NAME}")
    except Exception:
        print(f"Model {PADDLEOCR_MODEL_NAME} already exists — adding new version")

    model_uri = f"runs:/{run_id}/inference"
    mv = client.create_model_version(
        name=PADDLEOCR_MODEL_NAME,
        source=model_uri,
        run_id=run_id,
        description="Fine-tuned for tashkeel recognition + expanded vocabulary",
    )
    print(f"Registered {PADDLEOCR_MODEL_NAME} — version {mv.version}")

    client.transition_model_version_stage(
        name=PADDLEOCR_MODEL_NAME,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print(f"Set {PADDLEOCR_MODEL_NAME} v{mv.version} -> Staging "
          f"(promote to Production manually after CER review)")
    return mv.version
```

Modify `main()`:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--paddleocr-model-dir",
        default=None,
        help="Path to a fine-tuned PaddleOCR inference export to register (optional)",
    )
    args = parser.parse_args()

    print(f"Connecting to MLflow at: {args.tracking_uri}")
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(args.tracking_uri)

    yolo_version = register_yolo(client, args.tracking_uri)

    paddle_version = None
    if args.paddleocr_model_dir:
        paddle_version = register_paddleocr(client, args.tracking_uri, args.paddleocr_model_dir)

    print("\n=== Registration complete ===")
    print(f"YOLO model      : {YOLO_MODEL_NAME} v{yolo_version} (Production)")
    if paddle_version:
        print(f"PaddleOCR model : {PADDLEOCR_MODEL_NAME} v{paddle_version} (Staging)")
    print(f"View at: {args.tracking_uri}/#/models")
```

- [ ] **Step 2: Verify it runs (smoke test with --help)**

Run: `python scripts/register_model.py --help`
Expected: Shows both `--tracking-uri` and `--paddleocr-model-dir` options.

- [ ] **Step 3: Commit**

```bash
git add scripts/register_model.py
git commit -m "feat: register fine-tuned PaddleOCR model in MLflow (Staging stage)"
```

---

## Task 13: Create the Colab training notebook

**Files:**
- Create: `notebooks/05_finetune_paddleocr_arabic.ipynb`

- [ ] **Step 1: Create the notebook**

Create `notebooks/05_finetune_paddleocr_arabic.ipynb` with this content:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Fine-tune PaddleOCR Arabic Recognition — Tashkeel Enhancement\n",
    "\n",
    "Runs on Colab GPU (T4). Produces a fine-tuned `arabic_PP-OCRv5_mobile_rec` model that recognizes tashkeel (diacritics) in addition to plain Arabic text, with expanded vocabulary coverage and shadow-augmented training data.\n",
    "\n",
    "**Steps:** clone repos -> install deps -> generate synthetic data -> collect real crops -> merge -> train -> evaluate -> export -> download."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["!nvidia-smi"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 1. Clone repositories"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!git clone https://github.com/OmarElGohary/arabic-id-ocr.git\n",
    "%cd arabic-id-ocr\n",
    "!git checkout clean-pre-server"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["!git clone https://github.com/PaddlePaddle/PaddleOCR.git paddleocr_repo"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 2. Install dependencies"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -q mishkal arabic_reshaper python-bidi pillow opencv-python-headless\n",
    "!pip install -q -r paddleocr_repo/requirements.txt\n",
    "!pip install -q paddlepaddle-gpu==3.0.0"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Generate enhanced synthetic data\n",
    "\n",
    "Tashkeel via mishkal (~30% of samples), shadow augmentation, expanded vocabulary, alphanumeric serial format."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["!python scripts/generate_arabic_synthetic.py --count 20000 --out data/arabic_synthetic"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Collect real ID-card crops\n",
    "\n",
    "Upload the Egyptian ID dataset (Roboflow export) to `Egyptain-Person-ID-1/` before running this cell — same format used for YOLO training."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["!python scripts/collect_ocr_crops.py --dataset Egyptain-Person-ID-1 --out data/ocr_crops --min-conf 0.65"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 5. Merge into PaddleOCR training format"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["!python scripts/prepare_paddle_finetune.py --aug-factor 7 --val-ratio 0.1"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Download pretrained weights and fill in the config\n",
    "\n",
    "Starts fine-tuning from the current production `arabic_PP-OCRv5_mobile_rec` weights rather than from scratch."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import tarfile\n",
    "import urllib.request\n",
    "\n",
    "os.makedirs(\"pretrained\", exist_ok=True)\n",
    "url = \"https://paddleocr.bj.bcebos.com/PP-OCRv5/arabic/arabic_PP-OCRv5_mobile_rec_infer.tar\"\n",
    "urllib.request.urlretrieve(url, \"pretrained/arabic_PP-OCRv5_mobile_rec_infer.tar\")\n",
    "with tarfile.open(\"pretrained/arabic_PP-OCRv5_mobile_rec_infer.tar\") as t:\n",
    "    t.extractall(\"pretrained\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import yaml\n",
    "\n",
    "with open(\"configs/paddle_rec_finetune.yml\") as f:\n",
    "    cfg = yaml.safe_load(f)\n",
    "\n",
    "cfg[\"Global\"][\"pretrained_model\"] = \"pretrained/arabic_PP-OCRv5_mobile_rec_infer/arabic_PP-OCRv5_mobile_rec\"\n",
    "\n",
    "with open(\"configs/paddle_rec_finetune.yml\", \"w\") as f:\n",
    "    yaml.dump(cfg, f)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 7. Fine-tune"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%cd paddleocr_repo\n",
    "!python tools/train.py -c ../configs/paddle_rec_finetune.yml\n",
    "%cd .."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 8. Export the fine-tuned inference model"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%cd paddleocr_repo\n",
    "!python tools/export_model.py -c ../configs/paddle_rec_finetune.yml \\\n",
    "    -o Global.pretrained_model=../output/arabic_id_rec/best_accuracy \\\n",
    "       Global.save_inference_dir=../output/arabic_id_rec/inference/\n",
    "%cd .."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Evaluate: compare CER before vs after fine-tuning\n",
    "\n",
    "Reports overall CER plus a separate CER for the tashkeel-containing subset, for both the original and fine-tuned model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"=== Original model ===\")\n",
    "!python scripts/evaluate_cer.py --val-label data/paddle_finetune/val_label.txt --val-dir data/paddle_finetune/val --model-name arabic_PP-OCRv5_mobile_rec\n",
    "\n",
    "print(\"\\n=== Fine-tuned model ===\")\n",
    "!python scripts/evaluate_cer.py --val-label data/paddle_finetune/val_label.txt --val-dir data/paddle_finetune/val --model-dir output/arabic_id_rec/inference"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 10. Download the fine-tuned model"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import shutil\n",
    "from google.colab import files\n",
    "\n",
    "shutil.make_archive(\"arabic_id_rec_finetuned\", \"zip\", \"output/arabic_id_rec/inference\")\n",
    "files.download(\"arabic_id_rec_finetuned.zip\")"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
```

- [ ] **Step 2: Verify the notebook is valid JSON**

Run: `python -c "import json; json.load(open('notebooks/05_finetune_paddleocr_arabic.ipynb', encoding='utf-8'))"`
Expected: No error (confirms valid JSON / loadable notebook structure).

- [ ] **Step 3: Commit**

```bash
git add notebooks/05_finetune_paddleocr_arabic.ipynb
git commit -m "feat: add Colab notebook for PaddleOCR tashkeel fine-tuning"
```

---

## Task 14: Update CLAUDE.md to document the new fine-tuning pipeline

**Files:**
- Modify: `CLAUDE.md` (the "Fine-tuning OCR (Planned)" section)

- [ ] **Step 1: Add a note about the PaddleOCR-specific notebook**

In `CLAUDE.md`, in the `## Fine-tuning OCR (Planned)` section, add a new subsection right before the existing numbered TrOCR steps:

```markdown
## Fine-tuning OCR (Planned)

### PaddleOCR — tashkeel recognition + expanded vocabulary (current focus)

`notebooks/05_finetune_paddleocr_arabic.ipynb` fine-tunes the production
`arabic_PP-OCRv5_mobile_rec` model to recognize tashkeel (diacritics) and
expanded name/address/job vocabulary, using:

- `scripts/generate_arabic_synthetic.py` — synthetic data with mishkal-based
  tashkeel (~30% of samples), shadow/blur/skew/salt-and-pepper augmentation
- `scripts/collect_ocr_crops.py` — real ID-card crops auto-labeled with the
  current production model
- `scripts/prepare_paddle_finetune.py` — merges both into PaddleOCR training format
- `configs/paddle_rec_finetune.yml` — SVTR_LCNet training config
- `scripts/evaluate_cer.py` — compares CER of old vs fine-tuned model, with a
  separate score for the tashkeel subset
- `scripts/register_model.py --paddleocr-model-dir <path>` — registers the
  result in MLflow (Staging stage — promote to Production manually after CER review)

### TrOCR (alternative architecture, not yet pursued for production)
```

(The existing 5 numbered TrOCR steps stay unchanged below this, just under the renamed `### TrOCR` subheading instead of directly under `## Fine-tuning OCR (Planned)`.)

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document PaddleOCR tashkeel fine-tuning pipeline in CLAUDE.md"
```

---

## Task 15: Full test suite verification

**Files:** None (verification only)

- [ ] **Step 1: Run the complete test suite**

Run: `pytest tests/ -v`
Expected: All tests pass, including the new `tests/test_synthetic_generation.py`, `tests/test_evaluation.py`, and the additions to `tests/test_models.py` and `tests/test_api.py`.

- [ ] **Step 2: If any test fails, fix and re-run**

Investigate the specific failure, fix the implementation (not the test, unless the test itself has a bug), and re-run until green.

- [ ] **Step 3: Final commit (if any fixes were needed)**

```bash
git add -A
git commit -m "fix: address test failures found in full suite run"
```

(Skip this step if Step 1 passed cleanly with no changes needed.)
