"""Reference image / video -> style references (key-frames) + colour palette."""
import cv2
import numpy as np
from PIL import Image


def extract_keyframes(video_path: str, n: int = 4, max_side: int = 768):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    if total <= 0:
        return out
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
        ok, fr = cap.read()
        if ok:
            im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            im.thumbnail((max_side, max_side))
            out.append(im)
    cap.release()
    return out


def dominant_colors(images, k=4):
    px = []
    for im in images:
        a = np.asarray(im.convert("RGB").resize((64, 64)), np.float32).reshape(-1, 3)
        sat = a.max(1) - a.min(1)
        px.append(a[(sat > 40) & (a.max(1) > 60)])        # skip white/black/gray
    px = np.vstack(px) if px else np.zeros((0, 3), np.float32)
    if len(px) < k:
        return []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab, cen = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    order = np.argsort(-np.bincount(lab.ravel(), minlength=k))
    return [tuple(int(v) for v in cen[i]) for i in order]


def load_references(ref_images, ref_video):
    refs = []
    for p in (ref_images or []):
        try:
            refs.append(Image.open(p if isinstance(p, str) else p.name).convert("RGB"))
        except Exception:
            pass
    if ref_video:
        refs += extract_keyframes(ref_video if isinstance(ref_video, str) else ref_video.name)
    return refs
