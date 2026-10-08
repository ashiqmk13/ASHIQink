"""Download models after installation:  python -m inkreel.prefetch --profile cpu"""
import argparse
import os
from pathlib import Path

from . import DATA, hardware
from .langs import LANGS


def _dir_size(p: Path):
    t = 0
    for r, _, fs in os.walk(p):
        for f in fs:
            try:
                t += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return t


def disk_report():
    parts = []
    for sub in sorted(p for p in DATA.glob("*") if p.is_dir()):
        parts.append(f"{sub.name}: {_dir_size(sub) / 1e9:.2f} GB")
    from .config import FONTS_DIR
    parts.append(f"fonts: {_dir_size(FONTS_DIR) / 1e6:.1f} MB")
    return "  |  ".join(parts) or "empty"


def prefetch_llm(name, log=print):
    from .llm import LLM
    m = LLM(name, "auto", log); m.load(); m.unload()


def prefetch_image(key, log=print):
    from .imagegen import ImageGenerator
    g = ImageGenerator(key, "auto", None, log); g.load(); g.unload()


def prefetch_tts(engine, lang="en", log=print):
    from . import tts
    info = LANGS[lang]
    if not tts.engine_support(lang).get(engine):
        log(f"! {engine} does not support {info['name']}"); return
    eng = tts.get_engine(engine, hardware.resolve_device("auto"), log)
    ok, why = eng.available()
    if not ok:
        log(f"! {engine}: {why}"); return
    eng.synth("Hello." if lang.startswith("en") else "Hello world.", info)
    eng.close()
    log(f"  {engine} ready for {info['name']}")


def run(llm=None, image=None, tts=None, tts_lang="en", font_key=None, font_keys=None, log=print):
    from . import fonts
    keys = font_keys or font_key
    if keys:
        log("Fonts ..."); fonts.ensure_fonts(keys if isinstance(keys, list) else [keys], log)
    if llm and llm not in ("(skip)", "skip"):
        log(f"LLM {llm} ..."); prefetch_llm(llm, log)
    if image and image not in ("(skip)", "skip", "doodle"):
        log(f"Image model {image} ..."); prefetch_image(image, log)
    for e in (tts or []):
        log(f"Voice engine {e} ({tts_lang}) ..."); prefetch_tts(e, tts_lang, log)
    log("Downloads finished.  " + disk_report())


PROFILES = {
    "minimal": dict(font_keys=["caveat", "notosans"], tts=["kokoro"]),
    "cpu": dict(font_keys=["caveat", "patrickhand", "kalam", "notosans"], llm="Qwen2.5-1.5B (default)", image="sd15-lcm", tts=["kokoro"]),
    "gpu": dict(font_keys=["caveat", "patrickhand", "kalam", "notosans"], llm="Qwen2.5-7B (best quality)", image="sdxl-lcm", tts=["kokoro"]),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), default="cpu")
    ap.add_argument("--lang", default="en", help="voice language to warm up (e.g. hi, ml, es)")
    ap.add_argument("--fonts", nargs="*", help="extra font keys, e.g. notomalayalam nototamil")
    a = ap.parse_args()
    p = dict(PROFILES[a.profile])
    if a.fonts:
        p["font_keys"] = list(p["font_keys"]) + a.fonts
    run(tts_lang=a.lang, **p)
