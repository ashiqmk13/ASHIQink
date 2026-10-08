"""Font manager: bundled-after-download handwriting fonts, per-script Noto fonts, user uploads."""
import shutil
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, features

from .config import FONTS_DIR

_GF = "https://github.com/google/fonts/raw/main/ofl/"
_NOTO = "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/{n}/hinted/ttf/{n}-Regular.ttf"
_NOTO_BOLD = "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/{n}/hinted/ttf/{n}-Bold.ttf"


def _noto(n):
    return [_NOTO.format(n=n), _GF + f"{n.lower()}/{n}%5Bwdth%2Cwght%5D.ttf", _GF + f"{n.lower()}/{n}-Regular.ttf"]


# key -> (filename, [candidate urls], what it covers)
FONT_CATALOG = {
    "caveat":      ("Caveat.ttf",          [_GF + "caveat/Caveat%5Bwght%5D.ttf"], "Handwriting - Latin, Cyrillic"),
    "patrickhand": ("PatrickHand-Regular.ttf", [_GF + "patrickhand/PatrickHand-Regular.ttf"], "Handwriting - Latin"),
    "indieflower": ("IndieFlower-Regular.ttf", [_GF + "indieflower/IndieFlower-Regular.ttf"], "Handwriting - Latin"),
    "kalam":       ("Kalam-Regular.ttf",   [_GF + "kalam/Kalam-Regular.ttf"], "Handwriting - Latin + Devanagari (Hindi/Marathi)"),
    "notodeva":    ("NotoSansDevanagari-Regular.ttf", _noto("NotoSansDevanagari"), "Hindi / Marathi"),
    "notomalayalam": ("NotoSansMalayalam-Regular.ttf", _noto("NotoSansMalayalam"), "Malayalam"),
    "nototamil":   ("NotoSansTamil-Regular.ttf", _noto("NotoSansTamil"), "Tamil"),
    "nototelugu":  ("NotoSansTelugu-Regular.ttf", _noto("NotoSansTelugu"), "Telugu"),
    "notokannada": ("NotoSansKannada-Regular.ttf", _noto("NotoSansKannada"), "Kannada"),
    "notobengali": ("NotoSansBengali-Regular.ttf", _noto("NotoSansBengali"), "Bengali"),
    "notogujarati": ("NotoSansGujarati-Regular.ttf", _noto("NotoSansGujarati"), "Gujarati"),
    "notogurmukhi": ("NotoSansGurmukhi-Regular.ttf", _noto("NotoSansGurmukhi"), "Punjabi"),
    "notoarabic":  ("NotoNaskhArabic-Regular.ttf", _noto("NotoNaskhArabic"), "Arabic / Urdu"),
    "notothai":    ("NotoSansThai-Regular.ttf", _noto("NotoSansThai"), "Thai"),
    "notohebrew":  ("NotoSansHebrew-Regular.ttf", _noto("NotoSansHebrew"), "Hebrew"),
    "notosans":    ("NotoSans-Regular.ttf", _noto("NotoSans"), "Latin / Greek / Cyrillic / Vietnamese"),
    "notosc":      ("NotoSansCJKsc-Regular.otf", ["https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"], "Chinese (large ~16 MB)"),
    "notojp":      ("NotoSansCJKjp-Regular.otf", ["https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf"], "Japanese (large ~16 MB)"),
    "notokr":      ("NotoSansCJKkr-Regular.otf", ["https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/Korean/NotoSansCJKkr-Regular.otf"], "Korean (large ~16 MB)"),
}

SCRIPT_FONT = {
    "latin": "caveat", "cyrillic": "caveat", "devanagari": "kalam", "malayalam": "notomalayalam",
    "tamil": "nototamil", "telugu": "nototelugu", "kannada": "notokannada", "bengali": "notobengali",
    "gujarati": "notogujarati", "gurmukhi": "notogurmukhi", "arabic": "notoarabic", "thai": "notothai",
    "hebrew": "notohebrew", "greek": "notosans", "cjk_sc": "notosc", "cjk_jp": "notojp", "cjk_kr": "notokr",
}
BASIC_KEYS = ["caveat", "patrickhand", "indieflower", "kalam", "notosans"]


def _download(urls, dest: Path, log=print):
    for u in urls:
        try:
            log(f"  downloading font {dest.name} ...")
            req = urllib.request.Request(u, headers={"User-Agent": "InkReel/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(str(dest) + ".part", "wb") as f:
                shutil.copyfileobj(r, f)
            Path(str(dest) + ".part").replace(dest)
            return True
        except Exception as e:
            log(f"  ! {u.split('/')[-1]} failed: {e}")
    return False


def ensure_font(key: str, log=print):
    fn, urls, _ = FONT_CATALOG[key]
    dest = FONTS_DIR / fn
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    return dest if _download(urls, dest, log) else None


def ensure_fonts(keys=None, log=print):
    for k in (keys or BASIC_KEYS):
        ensure_font(k, log)


def list_fonts():
    return sorted(p.name for p in FONTS_DIR.iterdir() if p.suffix.lower() in (".ttf", ".otf", ".ttc"))


def add_font(path: str) -> str:
    p = Path(path)
    dest = FONTS_DIR / p.name
    shutil.copy(p, dest)
    return dest.name


def _fallback_font():
    try:
        import matplotlib
        p = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        if p.exists():
            return str(p)
    except Exception:
        pass
    return None


def resolve_font(name: str | None, script: str = "latin", log=print) -> str | None:
    """name = file in fonts/ (uploaded or downloaded). 'Auto' -> best font for the script."""
    if name and name not in ("Auto", "auto", ""):
        p = FONTS_DIR / name
        if p.exists():
            return str(p)
    key = SCRIPT_FONT.get(script, "caveat")
    fn = FONT_CATALOG[key][0]
    p = FONTS_DIR / fn
    if not p.exists():
        ensure_font(key, log)
    if p.exists():
        return str(p)
    for k in ("notosans", "caveat"):
        q = FONTS_DIR / FONT_CATALOG[k][0]
        if q.exists():
            log(f"! font for script '{script}' unavailable, using {q.name} (glyphs may be missing)")
            return str(q)
    log(f"! no font for script '{script}' - using fallback; download fonts in the Fonts tab")
    return _fallback_font()


def load_font(path, size):
    eng = ImageFont.Layout.RAQM if features.check("raqm") else ImageFont.Layout.BASIC
    if path:
        try:
            return ImageFont.truetype(path, size, layout_engine=eng)
        except Exception:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default(size)


def wrap_text(text, font, max_w, cjk=False):
    lines = []
    for para in text.split("\n"):
        tokens = list(para) if (cjk or " " not in para.strip()) else para.split(" ")
        sep = "" if (cjk or " " not in para.strip()) else " "
        cur = ""
        for t in tokens:
            trial = (cur + sep + t) if cur else t
            if font.getlength(trial) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur); cur = t
        lines.append(cur)
    return lines


def render_text_alpha(text, canvas_wh, box, font_path, align="center", max_size=None, min_size=14, line_gap=1.15):
    """Auto-fit `text` into box=(x,y,w,h); returns 'L' image (canvas sized, white text)."""
    W, H = canvas_wh
    bx, by, bw, bh = box
    cjk = any(ord(c) > 0x2E80 for c in text)
    size = int(max_size or bh)
    while True:
        f = load_font(font_path, size)
        lines = wrap_text(text, f, bw, cjk)
        asc, desc = f.getmetrics()
        lh = int((asc + desc) * line_gap)
        tw = max(f.getlength(l) for l in lines)
        if (tw <= bw and lh * len(lines) <= bh) or size <= min_size:
            break
        size = int(size * 0.92) - 1
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    y = by + (bh - lh * len(lines)) // 2
    for l in lines:
        w = f.getlength(l)
        x = bx + (0 if align == "left" else (bw - w) if align == "right" else (bw - w) / 2)
        d.text((x, y), l, font=f, fill=255)
        y += lh
    return img


def preview(font_name, text, script="latin", size=64):
    path = resolve_font(font_name, script)
    img = Image.new("RGB", (900, 220), "white")
    d = ImageDraw.Draw(img)
    d.text((20, 20), text, font=load_font(path, size), fill=(20, 20, 20))
    return img
