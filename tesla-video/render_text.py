"""Overlay the "Mi primer carro" captions on the Tesla video.

Usage: python3 render_text.py INPUT.mp4 OUTPUT.mp4 [--preview T1,T2,...]

Three type styles, no glow:
  body   – Montserrat ExtraBold, white
  bold   – Bebas Neue, light pink
  script – Brittany Signature, white (drop fonts/BrittanySignature.ttf or .otf
           in place; until then Mrs Saint Delafield stands in for it)
Every caption is timed to the actual cuts of the source video.
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))


def font_path(*names):
    for name in names:
        path = os.path.join(HERE, "fonts", name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(names[-1])


FONTS = {
    "body": font_path("Montserrat-800.ttf"),
    "bold": font_path("BebasNeue-400.ttf"),
    "script": font_path("BrittanySignature.ttf", "BrittanySignature.otf",
                        "MrsSaintDelafield-400.ttf"),
}

W, H, FPS = 1080, 1920, 30
MAX_W = 900  # keeps text clear of the TikTok side buttons

WHITE = (255, 255, 255)
LIGHT_PINK = (255, 194, 214)
SHADOW = (0, 0, 0)

COLORS = {"body": WHITE, "bold": LIGHT_PINK, "script": WHITE}
SIZES = {"body": 58, "bold": 190, "script": 215}
STROKE = {"body": 0, "bold": 0, "script": 3}

# (start, end, block centre y, lines). A line is (kind, text[, size, start]).
# Cut points of the source: 2.43 makeup, 13.80 app screen, 16.70 tower,
# 18.43 platform, 22.20 reflection, 24.23 outside, 28.37 interior,
# 31.50 wheel/road, 36.03 interior pan, 38.20 driving, 45.13 road/interior to the end.
BEATS = [
    # 0:00 hook – already driving
    (0.00, 2.70, 1240, [
        ("body", "Así se veía el momento"),
        ("body", "que llevaba tanto tiempo"),
        ("script", "imaginando"),
    ]),
    (2.80, 6.20, 1240, [
        ("body", "el día en que fui por"),
        ("bold", "MI PRIMER CARRO", 175),
    ]),
    # makeup
    (6.50, 10.00, 1240, [
        ("body", "Me arreglé"),
        ("script", "con calma"),
    ]),
    (10.25, 13.55, 1240, [
        ("body", "pero con el corazón"),
        ("bold", "A MIL", 230),
    ]),
    # Tesla on the app screen – first hint
    (13.90, 16.10, 1240, [
        ("body", "Ya sabía que iba por"),
        ("bold", "MI TESLA", 210),
    ]),
    # delivery tower
    (16.20, 18.40, 1240, [
        ("body", "pero nada me preparó para"),
        ("script", "lo que iba a sentir", 190),
    ]),
    # reveal on the platform (then a breath with only the car)
    (18.65, 20.80, 1300, [
        ("body", "Y entonces"),
        ("bold", "LO VI", 240),
    ]),
    # reflection in the glass
    (22.35, 24.90, 1240, [
        ("body", "Era simplemente"),
        ("script", "perfecto", 240),
    ]),
    # side, wheel, sky
    (25.20, 28.30, 1240, [
        ("script", "Blanco", 245),
        ("body", "como siempre lo quise"),
    ]),
    # interior / screen
    (28.60, 31.40, 1240, [
        ("body", "Después de tanto esperar"),
        ("bold", "POR FIN LLEGÓ", 185),
    ]),
    # steering wheel, road, palm trees
    (32.00, 37.70, 1240, [
        ("body", "Estaba viviendo el momento"),
        ("body", "que tanto"),
        ("script", "había soñado"),
    ]),
    # you driving – thank you
    (38.90, 42.70, 1240, [
        ("script", "Gracias, amor", 220),
        ("body", "por hacerlo realidad"),
    ]),
    (42.90, 47.80, 1240, [
        ("body", "y por sorprenderme"),
        ("bold", "CADA DÍA", 220),
    ]),
]

TEXT_IN = 0.45    # fade + rise for body and bold
SCRIPT_IN = 0.85  # handwriting wipe
STAGGER = 0.18
OUT = 0.30


def render_line(kind, text, size=None):
    """Return (premultiplied RGBA float array, baseline y, ink box) for one line."""
    size = size or SIZES[kind]
    stroke = STROKE[kind]
    while True:
        font = ImageFont.truetype(FONTS[kind], size)
        box = font.getbbox(text, anchor="ls", stroke_width=stroke)
        if box[2] - box[0] <= MAX_W or size < 30:
            break
        size -= 2
    pad = 30
    w, h = box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad
    origin = (pad - box[0], pad - box[1])
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text(origin, text, font=font, fill=255, anchor="ls",
                             stroke_width=stroke, stroke_fill=255)
    m = np.asarray(img, dtype=np.float32) / 255.0
    col = np.array(COLORS[kind], np.float32) / 255.0

    out = np.zeros((h, w, 4), np.float32)

    def over(rgb, a):
        out[..., :3] = rgb * a[..., None] + out[..., :3] * (1 - a[..., None])
        out[..., 3] = a + out[..., 3] * (1 - a)

    # crisp, tight drop shadow (no glow) so white/pink stay readable on bright shots
    sh = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
    sh = np.roll(np.asarray(sh, np.float32) / 255.0, 3, axis=0) * 0.45
    over(np.array(SHADOW, np.float32) / 255.0, sh)
    over(col, m)
    return out, origin[1], (box[1], box[3])


def layout(beat):
    start, end, cy, lines = beat
    items, y, prev = [], 0.0, None
    for i, line in enumerate(lines):
        kind, text = line[:2]
        size = line[2] if len(line) > 2 else None
        # the opening hook is fully on screen from frame 0 (cover + loop)
        t0 = line[3] if len(line) > 3 else start + (i * STAGGER if start > 0 else 0)
        img, base, (top, bottom) = render_line(kind, text, size)
        if prev is not None:
            if prev["kind"] == "body" and kind == "body":
                y += SIZES["body"] * 1.35
            else:  # stack on the ink so swashes never collide
                y += prev["bottom"] - top + (14 if kind == "body" or prev["kind"] == "body" else 8)
        items.append(dict(kind=kind, img=img, base=y, off=base, top=top, bottom=bottom,
                          t0=t0, end=end, instant=(start == 0.0)))
        prev = items[-1]
    mid = (items[0]["base"] + items[0]["top"] + items[-1]["base"] + items[-1]["bottom"]) / 2
    for it in items:
        h, w = it["img"].shape[:2]
        it["x"] = (W - w) // 2
        it["y"] = int(round(cy + it["base"] - mid - it["off"]))
    return items


def ease(t):
    return 1 - (1 - t) ** 3


def frame_alpha(it, t):
    """Global alpha, vertical offset and optional wipe mask for an item at time t."""
    if t < it["t0"] or t >= it["end"]:
        return None
    a, dy, wipe = 1.0, 0, None
    if not it["instant"]:
        if it["kind"] == "script":
            p = min(1.0, (t - it["t0"]) / SCRIPT_IN)
            a = min(1.0, p * 3)
            wipe = ease(p) if p < 1 else None
        else:
            p = min(1.0, (t - it["t0"]) / TEXT_IN)
            a, dy = ease(p), int(round((1 - ease(p)) * 22))
    if t > it["end"] - OUT:
        a *= max(0.0, (it["end"] - t) / OUT)
    return a, dy, wipe


def wipe_mask(shape, p):
    h, w = shape
    slant, feather = 0.35 * h, 70.0
    e = -slant - feather + p * (w + slant + 2 * feather)
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    return np.clip((e + slant * (1 - ys / h) - xs) / feather, 0, 1)


def compose(items, t):
    canvas = np.zeros((H, W, 4), np.float32)
    for it in items:
        fa = frame_alpha(it, t)
        if fa is None:
            continue
        a, dy, wipe = fa
        layer = it["img"] * a
        if wipe is not None:
            layer = layer * wipe_mask(layer.shape[:2], wipe)[..., None]
        h, w = layer.shape[:2]
        x0, y0 = it["x"], it["y"] + dy
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(W, x0 + w), min(H, y0 + h)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        src = layer[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        dst = canvas[cy0:cy1, cx0:cx1]
        dst[...] = src + dst * (1 - src[..., 3:4])
    alpha = canvas[..., 3:4]
    rgb = np.where(alpha > 1e-4, canvas[..., :3] / np.maximum(alpha, 1e-4), 0)
    return (np.concatenate([rgb, alpha], axis=2) * 255 + 0.5).clip(0, 255).astype(np.uint8)


def duration(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                   "-of", "default=nw=1:nk=1", path])
    return float(out)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    items = [it for beat in BEATS for it in layout(beat)]
    preview = None
    if len(sys.argv) > 3 and sys.argv[3] == "--preview":
        preview = [float(x) for x in sys.argv[4].split(",")]

    if preview:
        os.makedirs(dst, exist_ok=True)
        for t in preview:
            ov = compose(items, t)
            bg = subprocess.check_output([
                "ffmpeg", "-v", "error", "-ss", str(t), "-i", src, "-frames:v", "1",
                "-vf", f"scale={W}:{H}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            bg = np.frombuffer(bg, np.uint8).reshape(H, W, 3).astype(np.float32)
            a = ov[..., 3:4].astype(np.float32) / 255.0
            img = ov[..., :3] * a + bg * (1 - a)
            Image.fromarray(img.astype(np.uint8)).save(os.path.join(dst, f"t{t:05.2f}.png"))
        return

    n = int(math.ceil(duration(src) * FPS))
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-stats",
        "-i", src,
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-framerate", str(FPS), "-i", "-",
        "-filter_complex",
        f"[0:v]scale={W}:{H}:flags=lanczos,setsar=1[bg];[bg][1:v]overlay=0:0:format=auto[v]",
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-movflags", "+faststart", dst,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    blank = np.zeros((H, W, 4), np.uint8).tobytes()
    for i in range(n):
        t = i / FPS
        active = any(frame_alpha(it, t) is not None for it in items)
        proc.stdin.write(compose(items, t).tobytes() if active else blank)
    proc.stdin.close()
    sys.exit(proc.wait())


if __name__ == "__main__":
    main()
