"""YouTube-ready extras: thumbnail, description with chapters, tags."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import fonts


def fmt_ts(t):
    t = int(t)
    h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{h}:{m:02}:{s:02}" if h else f"{m}:{s:02}"


def chapters(scene_info, max_chapters=10):
    """scene_info = [(start_sec, title)], grouped so YouTube gets <= max_chapters, first at 0:00 (needs >= 3)."""
    n = len(scene_info)
    if n < 3:
        return []
    step = max(1, int(np.ceil(n / max_chapters)))
    out = []
    for i in range(0, n, step):
        t, name = scene_info[i]
        out.append(f"{fmt_ts(0 if i == 0 else t)} {name[:60] if name else f'Part {len(out) + 1}'}")
    return out


def make_thumbnail(frame_rgb: np.ndarray, title: str, font_path, out_path, size=(1280, 720)):
    img = Image.fromarray(frame_rgb).convert("RGB")
    s = max(size[0] / img.width, size[1] / img.height)
    img = img.resize((int(img.width * s) + 1, int(img.height * s) + 1), Image.LANCZOS)
    x, y = (img.width - size[0]) // 2, (img.height - size[1]) // 2
    img = img.crop((x, y, x + size[0], y + size[1])).filter(ImageFilter.GaussianBlur(0.4))
    # bold title band at the top
    band = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(band)
    d.rectangle([0, 0, size[0], int(size[1] * 0.34)], fill=(255, 214, 10, 235))
    img = Image.alpha_composite(img.convert("RGBA"), band)
    alpha = fonts.render_text_alpha(title, size, (int(size[0] * 0.05), int(size[1] * 0.025), int(size[0] * 0.9), int(size[1] * 0.29)),
                                    font_path, max_size=170)
    txt = Image.new("RGBA", size, (20, 20, 24, 255)); txt.putalpha(alpha)
    img = Image.alpha_composite(img, txt).convert("RGB")
    img.save(out_path, quality=92)
    return out_path


def description_text(meta: dict, chapter_lines):
    parts = [meta.get("description", "").strip()]
    if chapter_lines:
        parts += ["", "Chapters:"] + chapter_lines
    tags = meta.get("tags") or []
    if tags:
        parts += ["", " ".join("#" + t.replace(" ", "") for t in tags[:5])]
    parts += ["", "Made with InkReel (open-source, runs locally)."]
    return "\n".join(parts)
