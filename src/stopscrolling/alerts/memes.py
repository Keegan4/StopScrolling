"""Generate meme-style "get back to work" images with Pillow.

Everything here is drawn in code (cartoon faces, captions, cards), so there are no image
files to ship. Put your own images in ``assets/memes/`` and they are mixed in as well.
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "C:/Windows/Fonts/impact.ttf",
    "/Library/Fonts/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
PLAIN_FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

CAPTIONS = [
    ("PUT THE PHONE DOWN", "YOUR EXAMS ARE WATCHING"),
    ("ONE MORE VIDEO", "SAID EVERY STUDENT WHO FAILED"),
    ("I SEE YOU", "SCROLLING"),
    ("THE PHONE", "WILL STILL BE THERE LATER"),
    ("YOUR FUTURE SELF", "IS DISAPPOINTED"),
    ("NOTIFICATIONS CAN WAIT", "YOUR GRADES CAN'T"),
    ("BRO REALLY THOUGHT", "NOBODY WAS WATCHING"),
    ("STUDY NOW", "SCROLL LATER"),
    ("THAT'S NOT A TEXTBOOK", "THAT'S A PHONE"),
    ("FOCUS MODE", "ACTIVATED (BY FORCE)"),
    ("ERROR 404", "MOTIVATION NOT FOUND"),
    ("YOU SAID 5 MINUTES", "IT'S BEEN 5 MINUTES"),
    ("THE SYLLABUS", "IS NOT ON TIKTOK"),
    ("CLOSE THE APP", "OPEN THE BOOK"),
    ("WHEN YOU PICK UP YOUR PHONE", "AND THE WEBCAM SEES IT"),
]
SHOUTS = [
    "PUT. THE. PHONE. DOWN.",
    "GET BACK TO WORK",
    "STOP SCROLLING",
    "EYES ON THE BOOK",
    "LOCK IN.",
    "NOPE.",
    "NOT NOW.",
    "STUDY TIME",
    "PHONE = BAD",
    "FOCUS!!!",
]
EXPRESSIONS = ["angry", "disappointed", "shocked", "crying", "suspicious"]
LOUD_COLORS = [
    (230, 40, 40),
    (255, 120, 0),
    (250, 200, 0),
    (40, 170, 70),
    (30, 120, 230),
    (150, 60, 220),
    (230, 30, 140),
    (20, 20, 20),
]


def _first_existing(paths: list[str]) -> str | None:
    return next((p for p in paths if Path(p).exists()), None)


@lru_cache(maxsize=64)
def font(size: int, plain: bool = False) -> ImageFont.FreeTypeFont:
    path = _first_existing(PLAIN_FONT_CANDIDATES if plain else FONT_CANDIDATES)
    if path is None:
        return ImageFont.load_default(size)
    return ImageFont.truetype(path, size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, fnt, max_w: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and draw.textlength(trial, font=fnt) > max_w:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def fit_text(draw, text, max_w, max_h, max_size, min_size=12, plain=False):
    """Largest font size (and wrapped lines) that fits ``text`` into max_w x max_h."""
    for size in range(max_size, min_size - 1, -2):
        fnt = font(size, plain)
        lines = _wrap(draw, text, fnt, max_w)
        line_h = int(size * 1.1)
        if line_h * len(lines) <= max_h and all(draw.textlength(ln, font=fnt) <= max_w for ln in lines):
            return fnt, lines, line_h
    fnt = font(min_size, plain)
    return fnt, _wrap(draw, text, fnt, max_w), int(min_size * 1.1)


def draw_text_block(img, text, box, max_size, fill="white", stroke="black", align_v="center", plain=False):
    """Centred, wrapped text inside ``box`` = (x0, y0, x1, y1), meme-style outline."""
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    fnt, lines, line_h = fit_text(draw, text, x1 - x0, y1 - y0, max_size, plain=plain)
    total = line_h * len(lines)
    y = {"top": y0, "bottom": y1 - total}.get(align_v, y0 + (y1 - y0 - total) // 2)
    stroke_w = 0 if stroke is None else max(2, fnt.size // 14)
    for ln in lines:
        w = draw.textlength(ln, font=fnt)
        draw.text(
            (x0 + (x1 - x0 - w) / 2, y),
            ln,
            font=fnt,
            fill=fill,
            stroke_width=stroke_w,
            stroke_fill=stroke,
        )
        y += line_h


# --------------------------------------------------------------------------- faces


def draw_face(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, expression: str) -> None:
    lw = max(3, int(r / 18))
    skin = (255, 205, 40)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=skin, outline="black", width=lw)

    ex, ey, er = r * 0.36, cy - r * 0.15, r * 0.14
    if expression == "shocked":
        er = r * 0.19
    for side in (-1, 1):
        x = cx + side * ex
        if expression == "crying":
            draw.arc([x - er, ey - er, x + er, ey + er], 200, 340, fill="black", width=lw)
            draw.ellipse([x - r * 0.06, ey + r * 0.05, x + r * 0.06, ey + r * 0.55], fill=(80, 170, 255))
            continue
        draw.ellipse([x - er, ey - er, x + er, ey + er], fill="white", outline="black", width=lw)
        pr = er * (0.35 if expression == "shocked" else 0.5)
        px = x + (er * 0.45 if expression == "suspicious" else 0)
        draw.ellipse([px - pr, ey - pr, px + pr, ey + pr], fill="black")
        if expression in ("disappointed", "suspicious"):  # heavy upper eyelid
            draw.rectangle([x - er - 1, ey - er - 1, x + er + 1, ey - er * 0.1], fill=skin)
            draw.line([x - er, ey - er * 0.1, x + er, ey - er * 0.1], fill="black", width=lw)

        # Eyebrows
        bx0, bx1 = x - side * r * 0.22, x + side * r * 0.2
        by = ey - er - r * 0.12
        if expression == "angry":
            draw.line([bx1, by - r * 0.1, bx0, by + r * 0.06], fill="black", width=lw * 2)
        elif expression in ("disappointed", "crying"):
            draw.line([bx1, by + r * 0.04, bx0, by - r * 0.08], fill="black", width=lw * 2)
        elif expression == "shocked":
            draw.arc([x - er, by - r * 0.25, x + er, by + r * 0.05], 200, 340, fill="black", width=lw * 2)
        elif expression == "suspicious":
            lift = r * 0.1 if side > 0 else 0
            draw.line([x - er, by - lift, x + er, by - lift], fill="black", width=lw * 2)

    my = cy + r * 0.45
    if expression == "angry":
        draw.arc([cx - r * 0.4, my - r * 0.05, cx + r * 0.4, my + r * 0.45], 200, 340, fill="black", width=lw * 2)
    elif expression == "disappointed":
        draw.line([cx - r * 0.3, my + r * 0.05, cx + r * 0.3, my + r * 0.05], fill="black", width=lw * 2)
    elif expression == "shocked":
        draw.ellipse([cx - r * 0.18, my - r * 0.15, cx + r * 0.18, my + r * 0.3], fill=(90, 20, 20), outline="black", width=lw)
    elif expression == "crying":
        draw.chord([cx - r * 0.35, my - r * 0.05, cx + r * 0.35, my + r * 0.45], 180, 360, fill=(90, 20, 20), outline="black", width=lw)
    elif expression == "suspicious":
        draw.line([cx - r * 0.25, my + r * 0.08, cx + r * 0.3, my - r * 0.04], fill="black", width=lw * 2)


# --------------------------------------------------------------------------- templates


def face_meme(rng: random.Random) -> Image.Image:
    w = rng.randint(340, 440)
    h = int(w * rng.uniform(1.0, 1.2))
    bg = rng.choice(LOUD_COLORS)
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    for i in range(0, w + h, 40):  # sunburst-ish stripes
        draw.line([(i, 0), (i - h, h)], fill=tuple(min(255, c + 25) for c in bg), width=14)
    draw_face(draw, w / 2, h / 2, min(w, h) * 0.27, rng.choice(EXPRESSIONS))
    top, bottom = rng.choice(CAPTIONS)
    pad = int(w * 0.04)
    draw_text_block(img, top, (pad, pad, w - pad, int(h * 0.22)), int(h * 0.11), align_v="top")
    draw_text_block(img, bottom, (pad, int(h * 0.78), w - pad, h - pad), int(h * 0.11), align_v="bottom")
    return _border(img, "black", 4)


def shout_card(rng: random.Random) -> Image.Image:
    w, h = rng.randint(320, 460), rng.randint(180, 260)
    img = Image.new("RGB", (w, h), rng.choice(LOUD_COLORS[:-1]))
    draw = ImageDraw.Draw(img)
    stripe = 18
    for i in range(-h, w + h, stripe * 2):  # hazard tape border
        for y0, y1 in ((0, stripe), (h - stripe, h)):
            draw.polygon([(i, y1), (i + stripe, y0), (i + stripe * 2, y0), (i + stripe, y1)], fill="black")
    draw.rectangle([0, 0, w - 1, stripe - 1], outline=(250, 210, 0))
    draw_text_block(img, rng.choice(SHOUTS), (16, stripe + 8, w - 16, h - stripe - 8), 90)
    return _border(img, (250, 210, 0), 5)


def timer_card(seconds: float, rng: random.Random) -> Image.Image:
    w, h = 400, 260
    img = Image.new("RGB", (w, h), (20, 20, 20))
    draw_text_block(img, "YOU'VE BEEN ON YOUR PHONE FOR", (20, 14, w - 20, 80), 34)
    secs = int(seconds)
    t = f"{secs // 60}:{secs % 60:02d}" if secs >= 60 else f"{secs} SECONDS"
    draw_text_block(img, t, (20, 85, w - 20, 185), 80, fill=(255, 60, 60))
    draw_text_block(img, rng.choice(["AND COUNTING", "THE CLOCK IS TICKING", "WORTH IT?"]), (20, 195, w - 20, h - 14), 34)
    return _border(img, (255, 60, 60), 5)


def error_dialog(seconds: float, rng: random.Random) -> Image.Image:
    """A jokey 'error box' (deliberately cartoonish, not a real OS dialog)."""
    w, h = 420, 210
    title_color = rng.choice([(200, 30, 30), (120, 30, 200), (230, 90, 0)])
    img = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w, 34], fill=title_color)
    draw.text((12, 6), "StopScrolling.exe", font=font(18, plain=True), fill="white")
    draw.text((w - 30, 4), "X", font=font(22), fill="white")
    draw.ellipse([20, 56, 76, 112], fill=(220, 30, 30), outline="black", width=3)
    draw.text((36, 58), "!", font=font(46), fill="white")
    msg = rng.choice(
        [
            "Productivity.exe has stopped responding.",
            "Fatal error: brain cells lost to scrolling.",
            "Your attention span has crashed.",
            "Unauthorised phone usage detected.",
        ]
    )
    draw_text_block(img, msg, (92, 48, w - 16, 110), 26, fill="black", stroke=None, plain=True)
    draw_text_block(
        img, f"Phone time: {int(seconds)} s. Please return to studying.", (92, 112, w - 16, 140), 18,
        fill=(70, 70, 70), stroke=None, plain=True,
    )
    bx0, by0, bx1, by1 = w // 2 - 110, h - 56, w // 2 + 110, h - 16
    draw.rectangle([bx0, by0, bx1, by1], fill=(225, 225, 225), outline=title_color, width=3)
    draw_text_block(img, "GET BACK TO WORK", (bx0 + 6, by0 + 4, bx1 - 6, by1 - 4), 22, fill="black", stroke=None)
    return _border(img, title_color, 2)


def custom_meme(path: Path, max_side: int = 440) -> Image.Image | None:
    try:
        img = Image.open(path).convert("RGB")
    except OSError:
        return None
    img.thumbnail((max_side, max_side))
    return _border(img, "black", 3)


def _border(img: Image.Image, color, width: int) -> Image.Image:
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width - 1, img.height - 1], outline=color, width=width)
    return img


def list_custom_memes(folder: Path | None) -> list[Path]:
    if folder is None or not Path(folder).is_dir():
        return []
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def make_meme(rng: random.Random, seconds_on_phone: float, custom: list[Path] | None = None) -> Image.Image:
    """A random meme. Custom images (if any) make up about half of them."""
    if custom and rng.random() < 0.5:
        img = custom_meme(rng.choice(custom))
        if img is not None:
            return img
    kind = rng.choices(["face", "shout", "timer", "error"], weights=[5, 3, 1, 2])[0]
    if kind == "face":
        return face_meme(rng)
    if kind == "shout":
        return shout_card(rng)
    if kind == "timer":
        return timer_card(seconds_on_phone, rng)
    return error_dialog(seconds_on_phone, rng)
