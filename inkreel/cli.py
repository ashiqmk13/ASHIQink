"""Headless:  python -m inkreel.cli script.txt --lang en --style "Cartoon explainer" --res 1080p"""
import argparse
from pathlib import Path

from . import pipeline
from .config import ASPECTS, RESOLUTIONS
from .imagegen import IMAGE_MODELS
from .llm import LLM_MODELS
from .styles import STYLE_NAMES


def main():
    ap = argparse.ArgumentParser(description="InkReel - script to whiteboard video")
    ap.add_argument("script", help="text file with the script")
    ap.add_argument("--lang", default="auto"); ap.add_argument("--title", default="")
    ap.add_argument("--engine", default="auto", choices=["auto", "kokoro", "chatterbox", "mms", "espeak"])
    ap.add_argument("--voice", default="auto"); ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--style", default=STYLE_NAMES[1], choices=STYLE_NAMES)
    ap.add_argument("--negative", default=""); ap.add_argument("--style-extra", default="")
    ap.add_argument("--image-model", default="sd15-lcm", choices=list(IMAGE_MODELS))
    ap.add_argument("--llm", default="Qwen2.5-1.5B (default)", choices=list(LLM_MODELS))
    ap.add_argument("--aspect", default="16:9", choices=[k.split()[0] for k in ASPECTS])
    ap.add_argument("--res", default="720p", choices=[k.split()[0] for k in RESOLUTIONS])
    ap.add_argument("--hand", default="hand1"); ap.add_argument("--mode", default="draw", choices=["draw", "scanner", "chunks"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--captions", action="store_true"); ap.add_argument("--music")
    ap.add_argument("--ref-image", nargs="*", default=[]); ap.add_argument("--ref-video")
    ap.add_argument("--font", default="Auto")
    a = ap.parse_args()
    st = pipeline.Settings(
        script=Path(a.script).read_text(encoding="utf-8"), lang=a.lang, title=a.title, tts_engine=a.engine, voice=a.voice,
        speech_speed=a.speed, style=a.style, negative=a.negative, style_extra=a.style_extra, image_model=a.image_model,
        llm_model=a.llm, aspect=a.aspect, resolution=a.res, hand=a.hand, anim_mode=a.mode, device=a.device,
        captions=a.captions, music=a.music, ref_images=a.ref_image, ref_video=a.ref_video, font=a.font)
    meta = pipeline.analyze(st)
    out = pipeline.render(st, meta, meta["scenes"], progress=lambda f, d="": print(f"[{int(f * 100):3d}%] {d}"))
    print("\nDONE:", out["video"])


if __name__ == "__main__":
    main()
