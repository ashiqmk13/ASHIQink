import json
from functools import lru_cache
import numpy as np
from PIL import Image
from .config import HANDS_DIR

META = json.loads((HANDS_DIR / "hands.json").read_text())
HAND_CHOICES = [(v["name"], k) for k, v in META.items()] + [("No hand / no pen", "none"), ("Custom upload", "custom")]


class Hand:
    def __init__(self, rgb_pm, alpha, tip):
        self.rgb_pm = rgb_pm      # float32 HxWx3 premultiplied
        self.a = alpha            # float32 HxWx1
        self.tip = tip            # (x, y) pixels
        self.h, self.w = alpha.shape[:2]


@lru_cache(maxsize=16)
def _load(key, canvas_h, scale, custom_path, tipx, tipy):
    if key == "none":
        return None
    if key == "custom" and custom_path:
        im = Image.open(custom_path).convert("RGBA"); tip = (tipx, tipy)
        dscale = 0.5
    else:
        m = META.get(key, META["hand1"])
        im = Image.open(HANDS_DIR / m["file"]).convert("RGBA"); tip = m["tip"]
        dscale = m["default_scale"]
    target_h = max(40, int(canvas_h * (scale or dscale)))
    s = target_h / im.height
    im = im.resize((max(2, int(im.width * s)), target_h), Image.LANCZOS)
    arr = np.asarray(im, np.float32)
    a = arr[..., 3:4] / 255.0
    return Hand(arr[..., :3] * a, a, (tip[0] * im.width, tip[1] * im.height))


def load_hand(key, canvas_h, scale=None, custom_path=None, tip_frac=(0.5, 0.0)):
    return _load(key, canvas_h, scale, custom_path, tip_frac[0], tip_frac[1])


def blit(dst: np.ndarray, hand: Hand, px: float, py: float):
    """Alpha-blend the hand onto dst so its pen tip sits at (px, py)."""
    H, W = dst.shape[:2]
    x = int(round(px - hand.tip[0])); y = int(round(py - hand.tip[1]))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + hand.w), min(H, y + hand.h)
    if x1 <= x0 or y1 <= y0:
        return
    sa = hand.a[y0 - y:y1 - y, x0 - x:x1 - x]
    sp = hand.rgb_pm[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = dst[y0:y1, x0:x1]
    roi[:] = (roi * (1.0 - sa) + sp).astype(np.uint8)
