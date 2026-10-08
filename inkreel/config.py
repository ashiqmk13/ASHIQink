from pathlib import Path
from . import ROOT, DATA

ASSETS_DIR = ROOT / "assets"
HANDS_DIR = ASSETS_DIR / "hands"
FONTS_DIR = ROOT / "fonts"
PROJECTS_DIR = ROOT / "projects"
CHATTERBOX_VENV = ROOT / ".venv_chatterbox"

for _d in (DATA, FONTS_DIR, PROJECTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ASPECTS = {"16:9 (YouTube)": (16, 9), "9:16 (Shorts / Reels)": (9, 16), "1:1 (Square)": (1, 1)}
RESOLUTIONS = {"480p (draft)": 480, "720p": 720, "1080p (upload ready)": 1080}


def canvas_size(aspect: str, res: str):
    """Return (W, H), both even. `res` is the short side for landscape/square, long-side-based for portrait."""
    a, b = next((v for k, v in ASPECTS.items() if k == aspect or k.startswith(aspect.split()[0])), (16, 9))
    short = next((v for k, v in RESOLUTIONS.items() if k == res or k.startswith(str(res).split()[0])), 720)
    if a >= b:  # landscape / square -> height = short side
        h = short
        w = round(h * a / b)
    else:       # portrait -> width = short side
        w = short
        h = round(w * b / a)
    return w - w % 2, h - h % 2
