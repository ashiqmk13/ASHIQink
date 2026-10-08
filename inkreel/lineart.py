"""Turn pictures / text into 'Layers' the drawing engine can animate."""
from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image

THEMES = {
    # bg colour, ink colour, does colour fill make sense?
    "whiteboard": dict(bg=(255, 255, 255), ink=(24, 24, 28), color=True),
    "paper":      dict(bg=(246, 238, 220), ink=(40, 32, 28), color=True),
    "blackboard": dict(bg=(30, 52, 44),    ink=(240, 240, 232), color=False),
}


@dataclass
class Layer:
    ink_rgb: np.ndarray        # HxWx3  final look of the pen strokes (canvas sized)
    alpha: np.ndarray          # HxW u8 stroke strength (0 = nothing here)
    color_rgb: np.ndarray      # HxWx3  final look incl. colour fills
    footprint: np.ndarray      # HxW u8 region the colour brush may paint
    use_color: bool = True
    kind: str = "image"        # image | text
    order: str = "nearest"     # nearest | reading


def fit_into(img: Image.Image, box, upscale_max=3.0):
    """Resize `img` to fit inside box=(x,y,w,h) keeping aspect; returns (arr, x, y)."""
    bx, by, bw, bh = box
    s = min(bw / img.width, bh / img.height)
    s = min(s, upscale_max * max(1.0, 1.0))
    nw, nh = max(2, int(img.width * s)), max(2, int(img.height * s))
    arr = np.asarray(img.convert("RGB").resize((nw, nh), Image.LANCZOS))
    return arr, bx + (bw - nw) // 2, by + (bh - nh) // 2


def _otsu(gray):
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def ink_strength(rgb: np.ndarray, mode="auto", suppress_color=False):
    """Return (alpha float32 0..1, kind) where kind in {'drawing','photo'}."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    border = np.concatenate([gray[:3].ravel(), gray[-3:].ravel(), gray[:, :3].ravel(), gray[:, -3:].ravel()])
    white_bg = border.mean() > 225
    sat_all = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    colorful = (sat_all > 60).mean() > 0.08
    if mode == "auto":
        mode = "drawing" if (white_bg and (suppress_color or not colorful)) else "photo"
    if mode == "drawing":
        g = cv2.GaussianBlur(gray, (0, 0), 0.8)
        t = min(max(_otsu(g), 90.0), 200.0)
        a = np.clip((t + 10 - g.astype(np.float32)) / 55.0, 0, 1)
        a = a * a * (3 - 2 * a)
        if suppress_color:   # coloured fills are painted later by the brush - only dark neutral pixels are "ink"
            sat = rgb.max(axis=2).astype(np.float32) - rgb.min(axis=2).astype(np.float32)
            a = a * (1.0 - np.clip((sat - 45.0) / 70.0, 0, 1))
    else:  # photo / painting -> pencil sketch via colour-dodge
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (0, 0), max(2.0, gray.shape[1] / 160))
        sk = cv2.divide(gray, 255 - blur, scale=256).astype(np.float32)
        a = np.clip((235 - sk) / 120.0, 0, 1)
        a = np.clip(a * 1.35, 0, 1)
    return a.astype(np.float32), mode


def _clean_specks(a: np.ndarray, min_area: int):
    m = (a > 0.4).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    small = np.nonzero(stats[1:, cv2.CC_STAT_AREA] < min_area)[0] + 1
    if len(small):
        kill = np.isin(lab, small)
        a = a.copy(); a[kill] = 0
    return a


def _footprint(nonbg: np.ndarray, W: int):
    """Filled region covering drawing + interior (for the colour brush)."""
    k = max(3, int(W * 0.02)) | 1
    closed = cv2.morphologyEx(nonbg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fill = np.zeros_like(nonbg)
    cv2.drawContours(fill, cnts, -1, 255, thickness=-1)
    fill = cv2.dilate(fill, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return fill


def image_layer(img: Image.Image, canvas_wh, box, theme="whiteboard", use_color=True,
                ink_mode="auto", ink_rgb=None) -> Layer:
    W, H = canvas_wh
    th = THEMES[theme]
    bg = np.array(th["bg"], np.uint8)
    ink_col = np.array(ink_rgb or th["ink"], np.float32)
    arr, x, y = fit_into(img, box)
    h, w = arr.shape[:2]
    want_color = bool(use_color and th["color"])
    a, kind = ink_strength(arr, ink_mode, suppress_color=want_color)
    a = _clean_specks(a, max(6, int(0.00002 * W * H)))
    a_full = np.zeros((H, W), np.float32)
    a_full[y:y + h, x:x + w] = a
    ink = np.empty((H, W, 3), np.float32); ink[:] = bg
    ink = ink * (1 - a_full[..., None]) + ink_col * a_full[..., None]
    ink = ink.astype(np.uint8)
    col = np.empty((H, W, 3), np.uint8); col[:] = bg
    if want_color:
        patch = arr.astype(np.float32)
        if theme != "whiteboard":      # multiply so the picture's white becomes the paper colour
            patch = patch / 255.0 * bg.astype(np.float32)
        col[y:y + h, x:x + w] = patch.astype(np.uint8)
        diff = np.abs(arr.astype(np.int16) - 255).sum(axis=2) > 36
        nonbg = np.zeros((H, W), np.uint8)
        nonbg[y:y + h, x:x + w] = diff.astype(np.uint8) * 255
        nonbg = np.maximum(nonbg, (a_full > 0.3).astype(np.uint8) * 255)
    else:
        col = ink.copy()
        nonbg = (a_full > 0.1).astype(np.uint8) * 255
    foot = _footprint(nonbg, W)
    return Layer(ink_rgb=ink, alpha=(a_full * 255).astype(np.uint8), color_rgb=col, footprint=foot,
                 use_color=want_color, kind="image", order="nearest")


def text_layer(alpha_img: Image.Image, canvas_wh, theme="whiteboard", color=None) -> Layer:
    """`alpha_img` = 'L' image canvas-sized with white text on black."""
    W, H = canvas_wh
    th = THEMES[theme]
    bg = np.array(th["bg"], np.float32)
    ink_col = np.array(color or th["ink"], np.float32)
    a = np.asarray(alpha_img.convert("L"), np.float32) / 255.0
    ink = (bg * (1 - a[..., None]) + ink_col * a[..., None]).astype(np.uint8)
    foot = cv2.dilate((a > 0.05).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return Layer(ink_rgb=ink, alpha=(a * 255).astype(np.uint8), color_rgb=ink.copy(), footprint=foot,
                 use_color=False, kind="text", order="reading")


def hex_to_rgb(s, default=None):
    if not s:
        return default
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return default
