"""Overlay the "Mi primer carro" captions on the Tesla video.

Usage: python3 render_text.py INPUT.mp4 OUTPUT.mp4 [--preview T1,T2,...]

Style follows the reference TikTok: a thin high-contrast serif (Playfair Display)
paired with a big calligraphic script (Great Vibes): plain white serif and
pink-to-fuchsia script.
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
PINK = (255, 105, 185)
FUCHSIA = (232, 0, 132)
SHADOW = (0, 0, 0)

# Serif: plain white, no halo. Script: pink-to-fuchsia with a fuchsia glow.
# fill: solid colour or (top, bottom) gradient; glow: (colour, radius, strength)
STYLES = {
    "serif": dict(fill=WHITE, glow=None),
    "script": dict(fill=(PINK, FUCHSIA), glow=(FUCHSIA, 16, 0.9)),
}

SERIF_SIZE = 70
SCRIPT_SIZE = 150

# (start, end, block centre y, lines). A line is (kind, text[, size, start]).
# Cut points of the source: 2.43 makeup, 13.80 app screen, 16.70 tower,
# 18.43 platform, 22.20 reflection, 24.23 outside, 28.37 interior,
# 31.50 wheel/road, 36.03 interior pan, 38.20 driving, 45.13 road/interior to the end.
BEATS = [
    # 0:00 hook – already driving
    (0.00, 2.70, 1240, [
        ("serif", "Así se veía el momento"),
        ("serif", "que llevaba tanto tiempo"),
        ("script", "imaginando:"),
    ]),
    (2.80, 6.20, 1240, [
        ("serif", "el día en que fui por"),
        ("script", "Mi primer carro", 165),
    ]),
    # makeup
    (6.50, 13.55, 1240, [
        ("serif", "Llevaba muchísimo tiempo"),
        ("serif", "imaginando cómo sería"),
        ("script", "ese día…", 170),
    ]),
    # Tesla on the app screen – first hint
    (13.90, 16.10, 1240, [
        ("serif", "Y aunque yo ya sabía"),
        ("serif", "que iba por"),
        ("script", "mi Tesla,", 170),
    ]),
    # delivery tower
    (16.20, 18.40, 1240, [
        ("serif", "creo que todavía no entendía"),
        ("script", "lo especial"),
        ("serif", "que iba a sentirse."),
    ]),
    # reveal on the platform (then a breath with only the car)
    (18.65, 20.80, 1300, [
        ("serif", "Hasta que finalmente"),
        ("script", "lo vi ahí…", 170),
    ]),
    # reflection in the glass
    (22.35, 24.90, 1240, [
        ("serif", "Y sí… era exactamente como"),
        ("script", "lo había imaginado.", 135),
    ]),
    # side, wheel, sky
    (25.20, 28.30, 1240, [
        ("script", "Blanco,", 185),
        ("serif", "como siempre lo había querido."),
    ]),
    # interior / screen
    (28.60, 31.40, 1240, [
        ("serif", "Y después de tanto imaginarlo…"),
        ("serif", "finalmente estaba"),
        ("script", "sentada ahí."),
    ]),
    # steering wheel, road, palm trees
    (32.00, 37.70, 1240, [
        ("serif", "Y por fin estaba viviendo"),
        ("serif", "ese momento que tantas veces"),
        ("script", "había imaginado."),
    ]),
    # you driving – thank you
    (38.90, 42.70, 1240, [
        ("script", "Gracias, amor,", 165),
        ("serif", "por convertir mis sueños"),
        ("serif", "en realidad…"),
    ]),
    (42.90, 47.80, 1240, [
        ("serif", "y por sorprenderme"),
        ("script", "cada día.", 190),
    ]),
]

SERIF_IN = 0.45   # fade + rise
SCRIPT_IN = 0.85  # handwriting wipe
STAGGER = 0.18
OUT = 0.30


def blur(mask, radius):
    img = Image.fromarray((mask * 255).astype(np.uint8))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0


def render_line(kind, text, size=None):
    """Return (premultiplied RGBA float array, baseline y, ink box) for one line."""
    st = STYLES[kind]
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
    sh = np.clip(np.roll(np.roll(blur(m, 4), 3, axis=0), 2, axis=1) * 1.6, 0, 1) * 0.6
    over(np.array(SHADOW, np.float32) / 255.0, sh)
    if st["glow"]:
        gcol, grad, gstr = st["glow"]
        g = np.clip(blur(m, grad) * 2.2 * gstr, 0, 1) * 0.85
        over(np.array(gcol, np.float32) / 255.0, g)
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
