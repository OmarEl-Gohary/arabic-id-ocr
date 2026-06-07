"""
Generate synthetic Arabic text images for OCR fine-tuning.

Renders Arabic words/phrases in various fonts on realistic backgrounds,
then saves (image, label) pairs to data/arabic_synthetic/.

The model trained on this data will be robust to blur and poor camera
quality because augmentation is applied during generation.

Usage:
    python scripts/generate_arabic_synthetic.py --count 10000 --out data/arabic_synthetic

Then zip and upload to Colab for fine-tuning.
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance
import numpy as np

# ── Arabic vocabulary covering all ID card field types ────────────────────────

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

GENDER_VALUES = ["ذكر", "أنثى"]

RELIGION_VALUES = ["مسلم", "مسيحي", "مسلمة", "مسيحية"]

STATUS_VALUES = ["أعزب", "متزوج", "متزوجة", "مطلق", "أرمل", "أرملة"]

# Arabic-Indic digits for numbers / dates / IDs
ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"


def random_arabic_indic_number(length: int) -> str:
    return "".join(random.choice(ARABIC_INDIC) for _ in range(length))


def random_date() -> str:
    d = random_arabic_indic_number(2)
    m = random_arabic_indic_number(2)
    y = "١٩" + random_arabic_indic_number(2)
    return f"{d}/{m}/{y}"


def random_id() -> str:
    return random_arabic_indic_number(14)


def random_serial() -> str:
    return random_arabic_indic_number(random.randint(7, 9))


def sample_text() -> str:
    """Return a random Arabic text sample covering all field types."""
    category = random.choices(
        ["name", "address", "job", "gender", "religion", "status",
         "id", "date", "serial", "multi_word"],
        weights=[25, 20, 10, 5, 5, 5, 10, 10, 5, 5],
    )[0]

    if category == "name":
        n = random.randint(1, 3)
        return " ".join(random.choice(ARABIC_NAMES) for _ in range(n))
    elif category == "address":
        parts = random.sample(ARABIC_ADDRESSES, random.randint(1, 2))
        prefix = random_arabic_indic_number(random.randint(1, 2)) + " " if random.random() < 0.4 else ""
        return prefix + " ".join(parts)
    elif category == "job":
        return random.choice(ARABIC_JOBS)
    elif category == "gender":
        return random.choice(GENDER_VALUES)
    elif category == "religion":
        return random.choice(RELIGION_VALUES)
    elif category == "status":
        return random.choice(STATUS_VALUES)
    elif category == "id":
        return random_id()
    elif category == "date":
        return random_date()
    elif category == "serial":
        return random_serial()
    else:
        # Multi-word phrase
        return " ".join(random.choice(ARABIC_NAMES) for _ in range(random.randint(2, 4)))


# ── Image generation ──────────────────────────────────────────────────────────

def find_arabic_fonts() -> list:
    """Return a list of available Arabic TTF font paths."""
    candidates = [
        # Windows system fonts (Arabic support)
        r"C:\Windows\Fonts\trado.ttf",       # Traditional Arabic
        r"C:\Windows\Fonts\arabtype.ttf",    # Arabic Typesetting
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",     # Segoe UI — good Arabic coverage
        r"C:\Windows\Fonts\calibri.ttf",
        # macOS system fonts
        "/System/Library/Fonts/GeezaPro.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/opt/homebrew/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/arabic/amiri/Amiri-Regular.ttf",
        "/usr/share/fonts/truetype/amiri/Amiri-Regular.ttf",
    ]
    found = [p for p in candidates if os.path.exists(p)]
    if not found:
        print("WARNING: No Arabic fonts found. Using PIL default (quality will be low).")
        print("Windows: ensure trado.ttf / arabtype.ttf exist in C:\\Windows\\Fonts\\")
        print("Linux:   apt install fonts-amiri")
    return found


FONTS_CACHE: dict = {}

def get_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (font_path, size)
    if key not in FONTS_CACHE:
        FONTS_CACHE[key] = ImageFont.truetype(font_path, size)
    return FONTS_CACHE[key]


def make_background(w: int, h: int, with_watermark: bool = False) -> Image.Image:
    """Create a realistic ID-card-like background."""
    style = random.choice(["white", "light_color", "noisy_white"])

    if style == "white":
        bg = Image.new("RGB", (w, h), color=(255, 255, 255))
    elif style == "light_color":
        r = random.randint(200, 255)
        g = random.randint(200, 255)
        b = random.randint(200, 255)
        bg = Image.new("RGB", (w, h), color=(r, g, b))
    else:  # noisy_white
        arr = np.random.randint(230, 256, (h, w, 3), dtype=np.uint8)
        bg = Image.fromarray(arr)

    if with_watermark:
        # Simulate the security guilloche/watermark on Egyptian ID number fields.
        # Faint diagonal lines + dot grid mimics the real card background pattern.
        draw = ImageDraw.Draw(bg)
        shade = random.randint(195, 225)
        step  = random.randint(5, 9)
        for i in range(-(h), w + h, step):
            draw.line([(i, 0), (i + h, h)], fill=(shade, shade, shade + 5), width=1)
        for i in range(-(h), w + h, step * 2):
            draw.line([(i + h, 0), (i, h)], fill=(shade + 5, shade, shade), width=1)

    return bg


def augment(img: Image.Image) -> Image.Image:
    """Apply random degradation to simulate bad camera / scan quality."""

    # Gaussian blur (defocus / camera shake)
    if random.random() < 0.65:
        radius = random.uniform(0.3, 3.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Motion blur (horizontal camera movement)
    if random.random() < 0.35:
        kernel_size = random.choice([3, 5, 7, 9])
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
        arr = np.array(img)
        import cv2 as _cv2
        arr = _cv2.filter2D(arr, -1, kernel)
        img = Image.fromarray(arr)

    # Salt-and-pepper noise
    if random.random() < 0.35:
        arr     = np.array(img)
        density = random.uniform(0.01, 0.06)
        n_px    = int(arr.shape[0] * arr.shape[1] * density)
        for val in (255, 0):
            ys = np.random.randint(0, arr.shape[0], n_px)
            xs = np.random.randint(0, arr.shape[1], n_px)
            arr[ys, xs] = val
        img = Image.fromarray(arr)

    # Perspective skew (camera angle / card tilt)
    if random.random() < 0.35:
        w, h  = img.size
        skew  = random.uniform(-0.12, 0.12)
        coeffs = (1, skew, -skew * h / 2,
                  0, 1,    0,
                  0, 0)
        img = img.transform(
            (w, h), Image.PERSPECTIVE, coeffs,
            Image.BICUBIC, fillcolor=(255, 255, 255),
        )

    # Brightness
    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))

    # Contrast
    if random.random() < 0.45:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.55, 1.55))

    # Sharpness (defocus)
    if random.random() < 0.3:
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.1, 1.0))

    # Gaussian noise
    if random.random() < 0.35:
        arr   = np.array(img).astype(np.float32)
        sigma = random.uniform(5, 30)
        noise = np.random.normal(0, sigma, arr.shape)
        arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img   = Image.fromarray(arr)

    # Rotation (camera tilt)
    if random.random() < 0.4:
        angle = random.uniform(-7, 7)
        img   = img.rotate(angle, fillcolor=(255, 255, 255), expand=False)

    # JPEG compression artefacts
    if random.random() < 0.35:
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(30, 75))
        buf.seek(0)
        img = Image.open(buf).copy()

    return img


def reshape_arabic(text: str) -> str:
    """Reshape Arabic text so PIL renders connected, properly joined letters."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except ImportError:
        return text   # fall back if libraries not installed


def render_text_image(
    text: str,
    font_path: str,
    target_h: int = 64,
    min_w: int = 80,
    padding: int = 10,
    with_watermark: bool = False,
) -> Image.Image:
    """Render Arabic text onto a background image."""
    display_text = reshape_arabic(text)

    font_size = int(target_h * 0.65)
    font = get_font(font_path, font_size)

    dummy = Image.new("RGB", (1, 1))
    draw  = ImageDraw.Draw(dummy)
    bbox  = draw.textbbox((0, 0), display_text, font=font)
    tw    = bbox[2] - bbox[0]
    th    = bbox[3] - bbox[1]

    w = max(tw + padding * 2, min_w)
    h = th + padding * 2

    bg   = make_background(w, h, with_watermark=with_watermark)
    draw = ImageDraw.Draw(bg)

    r = random.randint(0, 60)
    g = random.randint(0, 60)
    b = random.randint(0, 60)
    x = (w - tw) // 2
    y = (h - th) // 2
    draw.text((x, y), display_text, font=font, fill=(r, g, b))

    return bg


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--count",      type=int, default=10000,
                   help="Number of synthetic images to generate")
    p.add_argument("--out",        default="data/arabic_synthetic",
                   help="Output directory")
    p.add_argument("--height",     type=int, default=64,
                   help="Target text height in pixels before augmentation")
    p.add_argument("--field-type", default="all",
                   choices=["all", "id", "address", "name", "numeric"],
                   help="Generate only samples for a specific field type")
    return p.parse_args()


# Field-type → sample_text category weights
_FIELD_WEIGHTS = {
    "id":      {"id": 1},
    "address": {"address": 1},
    "name":    {"name": 1, "multi_word": 1},
    "numeric": {"id": 2, "date": 2, "serial": 1},
    "all":     None,  # uses default weights in sample_text()
}


def main():
    args = parse_args()

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.csv"

    fonts = find_arabic_fonts()
    if not fonts:
        sys.exit(1)
    print(f"Found {len(fonts)} font(s): {[Path(f).name for f in fonts]}")
    print(f"Generating {args.count} images (field_type={args.field_type}) → {out_dir}\n")

    field_type   = args.field_type
    use_watermark_for_id = field_type in ("id", "numeric", "all")

    rows = []
    for i in range(args.count):
        # For targeted generation, force the category
        if field_type == "id":
            text = random_id()
        elif field_type == "address":
            parts = random.sample(ARABIC_ADDRESSES, random.randint(1, 2))
            prefix = random_arabic_indic_number(random.randint(1, 2)) + " " if random.random() < 0.4 else ""
            text = prefix + " ".join(parts)
        elif field_type == "name":
            text = " ".join(random.choice(ARABIC_NAMES) for _ in range(random.randint(1, 3)))
        elif field_type == "numeric":
            text = random.choice([random_id, random_date, random_serial])()
        else:
            text = sample_text()

        font_path    = random.choice(fonts)
        with_wm      = use_watermark_for_id and (field_type == "id" or
                        (field_type == "all" and random.random() < 0.12))

        try:
            img = render_text_image(text, font_path, target_h=args.height,
                                    with_watermark=with_wm)
            img = augment(img)
        except Exception:
            continue

        filename = f"syn_{i:06d}.png"
        img.save(img_dir / filename)
        rows.append({"filename": filename, "text": text})

        if (i + 1) % 1000 == 0 or (i + 1) == args.count:
            print(f"  {i+1}/{args.count}")

    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} images saved to {out_dir}")
    print(f"Labels: {labels_path}")


if __name__ == "__main__":
    main()
