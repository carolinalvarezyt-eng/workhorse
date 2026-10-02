"""Overlay the "Mi primer carro" captions on the Tesla video.

Usage: python3 render_text.py INPUT.mp4 OUTPUT.mp4 [--preview T1,T2,...]

Style follows the reference TikTok: a thin high-contrast serif (Playfair Display)
paired with a big calligraphic script (Great Vibes), in white, pink and fuchsia.
Every caption is timed to the actual cuts of the source video.
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SERIF = os.path.join(HERE, "fonts", "PlayfairDisplay-500.ttf")
SCRIPT = os.path.join(HERE, "fonts", "GreatVibes-400.ttf")

W, H, FPS = 1080, 1920, 30
MAX_W = 900  # keeps text clear of the TikTok side buttons

WHITE = (255, 255, 255)
PINK = (255, 190, 220)
PINK_LIGHT = (255, 222, 236)
HOT_PINK = (255, 95, 175)
FUCHSIA = (236, 0, 140)
SHADOW = (60, 0, 32)

# fill: solid colour or (top, bottom) gradient; glow: (colour, radius, strength)
STYLES = {
    "serif_white": dict(fill=WHITE, glow=(PINK, 8, 0.35)),
    "serif_pink": dict(fill=PINK, glow=(FUCHSIA, 8, 0.4)),
    "script_white": dict(fill=WHITE, glow=(FUCHSIA, 16, 1.0)),
    "script_pink": dict(fill=(PINK_LIGHT, HOT_PINK), glow=(FUCHSIA, 16, 0.9)),
    "script_fuchsia": dict(fill=(HOT_PINK, FUCHSIA), glow=(WHITE, 14, 0.75)),
}

SERIF_SIZE = 70
SCRIPT_SIZE = 150

# (start, end, block centre y, lines). A line is (kind, text, style[, size, start]).
# Cut points of the source: 11.40 app screen, 14.30 tower, 16.07 platform,
# 19.83 reflection, 21.83 outside, 26.00 interior, 29.13 wheel/road,
# 33.67 interior pan, 35.80 driving, 42.77 road/interior to the end.
BEATS = [
    # 0:00 hook – already in the car
    (0.00, 2.55, 1240, [
        ("serif", "Así se veía el momento", "serif_white"),
        ("serif", "que llevaba tanto tiempo", "serif_white"),
        ("script", "imaginando:", "script_pink"),
    ]),
    (2.60, 5.35, 1240, [
        ("serif", "el día en que fui por", "serif_pink"),
        ("script", "Mi primer carro", "script_fuchsia", 165),
    ]),
    # makeup
    (5.60, 11.20, 1240, [
        ("serif", "Llevaba muchísimo tiempo", "serif_white"),
        ("serif", "imaginando cómo sería", "serif_white"),
        ("script", "ese día…", "script_white", 170),
    ]),
    # Tesla on the app screen – first hint
    (11.50, 13.70, 1240, [
        ("serif", "Y aunque yo ya sabía", "serif_white"),
        ("serif", "que iba por", "serif_white"),
        ("script", "mi Tesla,", "script_fuchsia", 170),
    ]),
    # delivery tower
    (13.80, 16.00, 1240, [
        ("serif", "creo que todavía no entendía", "serif_pink"),
        ("script", "lo especial", "script_white"),
        ("serif", "que iba a sentirse.", "serif_pink"),
    ]),
    # reveal on the platform (then a breath with only the car)
    (16.25, 18.40, 1300, [
        ("serif", "Hasta que finalmente", "serif_white"),
        ("script", "lo vi ahí…", "script_pink", 170),
    ]),
    # reflection in the glass
    (19.95, 22.50, 1240, [
        ("serif", "Y sí… era exactamente como", "serif_pink"),
        ("script", "lo había imaginado.", "script_white", 135),
    ]),
    # side, wheel, sky
    (22.80, 25.90, 1240, [
        ("script", "Blanco,", "script_white", 185),
        ("serif", "como siempre lo había querido.", "serif_pink"),
    ]),
    # interior / screen
    (26.20, 29.00, 1240, [
        ("serif", "Y después de tanto imaginarlo…", "serif_white"),
        ("serif", "finalmente estaba", "serif_white"),
        ("script", "sentada ahí.", "script_fuchsia"),
    ]),
    # steering wheel, road, palm trees
    (29.35, 31.60, 1240, [
        ("serif", "Mi primer carro.", "serif_white"),
        ("script", "Mi Tesla.", "script_pink", 185),
    ]),
    (31.75, 35.60, 1240, [
        ("serif", "Y por fin estaba viviendo", "serif_pink"),
        ("serif", "ese momento que tantas veces", "serif_pink"),
        ("script", "había imaginado.", "script_white"),
    ]),
    # you driving – close
    (36.90, 44.20, 1240, [
        ("serif", "Y creo que fue ahí", "serif_white"),
        ("serif", "cuando entendí:", "serif_white"),
        ("script", "ya era mío.", "script_fuchsia", 190, 39.40),
    ]),
]

SERIF_IN = 0.45   # fade + rise
SCRIPT_IN = 0.85  # handwriting wipe
STAGGER = 0.18
OUT = 0.30


def blur(mask, radius):
    img = Image.fromarray((mask * 255).astype(np.uint8))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0


def render_line(kind, text, style, size=None):
    """Return (premultiplied RGBA float array, baseline y, ink box) for one line."""
    st = STYLES[style]
    size = size or (SCRIPT_SIZE if kind == "script" else SERIF_SIZE)
    path = SCRIPT if kind == "script" else SERIF
    stroke = 2 if kind == "script" else 0  # the reference script is heavier
    while True:
        font = ImageFont.truetype(path, size)
        box = font.getbbox(text, anchor="ls", stroke_width=stroke)
        if box[2] - box[0] <= MAX_W or size < 40:
            break
        size -= 2
    pad = 60
    w, h = box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad
    origin = (pad - box[0], pad - box[1])
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text(origin, text, font=font, fill=255, anchor="ls",
                             stroke_width=stroke, stroke_fill=255)
    m = np.asarray(img, dtype=np.float32) / 255.0

    fill = st["fill"]
    if isinstance(fill[0], tuple):
        top, bot = np.array(fill[0], np.float32), np.array(fill[1], np.float32)
        t = np.clip((np.arange(h) - pad) / max(1, h - 2 * pad), 0, 1)[:, None, None]
        col = np.broadcast_to(top + (bot - top) * t, (h, w, 3)) / 255.0
    else:
        col = np.broadcast_to(np.array(fill, np.float32) / 255.0, (h, w, 3))

    out = np.zeros((h, w, 4), np.float32)

    def over(rgb, a):
        out[..., :3] = rgb * a[..., None] + out[..., :3] * (1 - a[..., None])
        out[..., 3] = a + out[..., 3] * (1 - a)

    # soft drop shadow for legibility on bright shots
    sh = np.clip(np.roll(np.roll(blur(m, 5), 3, axis=0), 2, axis=1) * 1.6, 0, 1) * 0.7
    over(np.array(SHADOW, np.float32) / 255.0, sh)
    gcol, grad, gstr = st["glow"]
    g = np.clip(blur(m, grad) * 2.2 * gstr, 0, 1) * 0.85
    over(np.array(gcol, np.float32) / 255.0, g)
    over(col, m)
    return out, origin[1], (box[1], box[3])


def layout(beat):
    start, end, cy, lines = beat
    items, y, prev = [], 0.0, None
    for i, line in enumerate(lines):
        kind, text, style = line[:3]
        size = line[3] if len(line) > 3 else None
        # the opening hook is fully on screen from frame 0 (cover + loop)
        t0 = line[4] if len(line) > 4 else start + (i * STAGGER if start > 0 else 0)
        img, base, (top, bottom) = render_line(kind, text, style, size)
        if prev is not None:
            if prev["kind"] == "serif" and kind == "serif":
                y += SERIF_SIZE * 1.3
            else:  # stack on the ink so swashes never collide
                y += prev["bottom"] - top + (6 if kind == "script" else 16)
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
        if it["kind"] == "serif":
            p = min(1.0, (t - it["t0"]) / SERIF_IN)
            a, dy = ease(p), int(round((1 - ease(p)) * 22))
        else:
            p = min(1.0, (t - it["t0"]) / SCRIPT_IN)
            a = min(1.0, p * 3)
            wipe = ease(p) if p < 1 else None
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
