"""Orchestrates: script -> scenes -> voice -> images -> whiteboard animation -> final YouTube-ready MP4."""
import json
import math
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

from . import fonts, hardware, hands, lineart, reference, styles, tts, video, youtube
from .config import PROJECTS_DIR, canvas_size
from .drawing import render_scene
from .imagegen import IMAGE_MODELS, ImageGenerator, cache_key, gen_size
from .langs import LANGS, detect_lang
from .llm import LLM, LLM_MODELS, analyze_script
from .script_utils import split_scenes, short_label
from .tts import SR


class Cancelled(Exception):
    pass


@dataclass
class Settings:
    # script
    script: str = ""
    title: str = ""
    lang: str = "auto"
    translate_to: str = "none"
    scene_sec: float = 9.0
    llm_model: str = "Qwen2.5-1.5B (default)"
    label_mode: str = "ai"                 # ai | first words | none
    # voice
    tts_engine: str = "auto"
    voice: str = "auto"
    speech_speed: float = 1.0
    ref_voice: str | None = None
    # visuals
    style: str = styles.STYLE_NAMES[1]
    style_extra: str = ""
    negative: str = ""
    image_model: str = "sd15-lcm"
    custom_checkpoint: str = ""
    image_steps: int = 0
    seed: int = 1234
    ref_images: list = field(default_factory=list)
    ref_video: str | None = None
    ref_strength: float = 0.6
    user_images: list = field(default_factory=list)
    use_color: str = "style"               # style | yes | no
    # animation
    anim_mode: str = "draw"                # draw | scanner | chunks
    hand: str = "hand1"
    hand_scale: float = 0.0
    custom_hand: str | None = None
    custom_hand_tip: tuple = (0.5, 0.0)
    draw_ratio: float = 0.8
    text_first: bool = True
    font: str = "Auto"
    text_color: str = ""
    zoom: bool = True
    transition: str = "wipe"               # wipe | none
    captions: bool = False
    # video
    aspect: str = "16:9 (YouTube)"
    resolution: str = "720p"
    fps: int = 30
    device: str = "auto"
    music: str | None = None
    music_vol: float = 0.12


def _check(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def new_project(title="") -> Path:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-")[:30] or "video"
    d = PROJECTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}"
    for s in ("audio", "images", "segments"):
        (d / s).mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------- stage 1: analysis
def analyze(st: Settings, log=print, cancel=None):
    script = st.script.strip()
    if not script:
        raise ValueError("Script is empty")
    lang = detect_lang(script) if st.lang == "auto" else st.lang
    log(f"Language: {LANGS[lang]['name']}")
    narrations = split_scenes(script, lang, st.scene_sec)
    log(f"Split into {len(narrations)} scene(s)")
    llm = None
    spec = LLM_MODELS.get(st.llm_model)
    if spec:
        llm = LLM(st.llm_model, st.device, log)
    translate_to = None if st.translate_to in ("none", "", None) else st.translate_to
    try:
        meta = analyze_script(narrations, lang, llm, st.label_mode, translate_to, log, cancel)
    except InterruptedError:
        raise Cancelled()
    finally:
        if llm:
            llm.unload()
    if st.title.strip():
        meta["title"] = st.title.strip()
    log("Analysis done - review / edit the scenes, then generate the video.")
    return meta


# ----------------------------------------------------------------------------- layout helpers
def layout(W, H, has_text, has_image, captions):
    portrait = H > W
    mx = 0.05 * W
    bottom = 0.15 * H if captions else 0.04 * H
    top = 0.03 * H
    if has_text and has_image:
        th = (0.12 if portrait else 0.17) * H
        text_box = (mx, top, W - 2 * mx, th)
        y0 = top + th + 0.02 * H
        img_box = (0.04 * W, y0, 0.92 * W, H - bottom - y0)
    elif has_image:
        text_box = None
        img_box = (0.04 * W, 0.04 * H, 0.92 * W, H - bottom - 0.04 * H)
    else:
        text_box = (mx, 0.28 * H, W - 2 * mx, 0.4 * H)
        img_box = None
    ints = lambda b: tuple(int(v) for v in b) if b else None
    return ints(text_box), ints(img_box)


def _scene_layers(st, W, H, theme, use_color, text, image, font_path, ink_rgb, text_rgb):
    tb, ib = layout(W, H, bool(text), image is not None, st.captions)
    layers = []
    tl = il = None
    if text and tb:
        a = fonts.render_text_alpha(text, (W, H), tb, font_path, max_size=int(tb[3] * 0.95))
        tl = lineart.text_layer(a, (W, H), theme, text_rgb)
    if image is not None and ib:
        il = lineart.image_layer(image, (W, H), ib, theme, use_color, "auto", ink_rgb)
    layers = [x for x in ((tl, il) if st.text_first else (il, tl)) if x is not None]
    return layers


def _render_segment(st, W, H, theme, bg, layers, hand, seg_path, speech_sec, fps, is_last, overlay_factory, cancel, seed):
    lead = 0.35
    wipe = (st.transition == "wipe") and not is_last
    wipe_sec = 0.55 if wipe else 0.0
    tail = 0.5 + wipe_sec
    n_frames = int(math.ceil((lead + speech_sec + tail) * fps))
    total = n_frames / fps
    draw_start, exit_dur = 0.30, 0.40
    draw_dur = st.draw_ratio * speech_sec
    draw_dur = max(1.0, min(draw_dur, total - wipe_sec - 0.25 - exit_dur - draw_start))
    wr = SegmentWriter = video.SegmentWriter(seg_path, W, H, fps)
    try:
        gen = render_scene(layers, bg, W, H, fps, n_frames, draw_start, draw_dur, exit_dur, hand,
                           anim_mode=st.anim_mode, zoom=st.zoom, wipe_frames=int(wipe_sec * fps),
                           overlay=overlay_factory(lead, speech_sec) if overlay_factory else None, seed=seed, cancel=cancel)
        for f in gen:
            wr.write(f)
        _check(cancel)
    except BaseException:
        wr.abort()
        raise
    wr.close()
    return n_frames, total, lead


# ----------------------------------------------------------------------------- stage 2: full render
def render(st: Settings, meta: dict, scenes: list, project: Path | None = None, log=print, progress=lambda f, d="": None,
           cancel=None):
    """scenes: list of dict(narration, image_prompt, on_screen_text). Returns dict of output paths."""
    t_start = time.time()
    project = Path(project) if project else new_project(meta.get("title", ""))
    for s in ("audio", "images", "segments"):
        (project / s).mkdir(parents=True, exist_ok=True)
    scenes = [s for s in scenes if (s.get("narration") or "").strip()]
    if not scenes:
        raise ValueError("No scenes to render")
    n = len(scenes)
    lang = meta.get("lang") or (detect_lang(st.script) if st.lang == "auto" else st.lang)
    device = hardware.resolve_device(st.device)
    W, H = canvas_size(st.aspect, st.resolution)
    fps = int(st.fps)
    style = styles.STYLES[st.style]
    theme = style["theme"]
    bg = lineart.THEMES[theme]["bg"]
    use_color = style["color"] if st.use_color == "style" else (st.use_color == "yes")
    script_kind = LANGS[lang]["script"]
    font_path = fonts.resolve_font(st.font, script_kind, log)
    from PIL import features as _pf
    if script_kind in ("devanagari", "malayalam", "tamil", "telugu", "kannada", "bengali", "gujarati", "gurmukhi", "arabic", "thai") and not _pf.check("raqm"):
        log("! Pillow has no RAQM text-shaping: Indic/Arabic letters may look broken. Fix: Linux `sudo apt install libraqm0`, "
            "macOS `brew install libraqm`, Windows `pip install --upgrade pillow`.")
    (project / "scenes.json").write_text(json.dumps(dict(meta=meta, scenes=scenes, settings=asdict(st)), ensure_ascii=False, indent=1, default=str))
    log(f"Project folder: {project}")
    log(f"Canvas {W}x{H} @ {fps} fps | device: {device} | style: {st.style}")

    # ---------------- A. voice-over
    progress(0.02, "Voice-over ...")
    engine = tts.pick_engine(st.tts_engine, lang, device, log)
    log(f"Voice engine: {engine.name}")
    wavs = []
    for i, sc in enumerate(scenes):
        _check(cancel)
        key = cache_key(engine.name, lang, st.voice, st.speech_speed, sc["narration"], st.ref_voice or "")
        p = project / "audio" / f"{i:03}_{key}.wav"
        if p.exists():
            x, _ = sf.read(str(p), dtype="float32")
        else:
            log(f"  voice {i + 1}/{n} ...")
            x = engine.synth(sc["narration"], LANGS[lang], st.voice, st.speech_speed if getattr(engine, "native_speed", False) else 1.0, st.ref_voice)
            x = tts.trim_and_normalize(x)
            if not getattr(engine, "native_speed", False):
                x = video.change_tempo(x, SR, st.speech_speed)
            sf.write(str(p), x, SR)
        wavs.append(x)
        progress(0.02 + 0.18 * (i + 1) / n, f"Voice {i + 1}/{n}")
    engine.close(); hardware.free_memory()

    # ---------------- B. images
    imgs = [None] * n
    user_imgs = sorted([u if isinstance(u, str) else u.name for u in (st.user_images or [])])
    need_ai = [i for i, s in enumerate(scenes) if i >= len(user_imgs) and (s.get("image_prompt") or "").strip()]
    refs = reference.load_references(st.ref_images, st.ref_video)
    palette = styles.palette_words(reference.dominant_colors(refs)) if refs else ""
    for i in range(n):
        if i < len(user_imgs):
            imgs[i] = Image.open(user_imgs[i]).convert("RGB")
    if need_ai:
        gen = ImageGenerator(st.image_model, device, st.custom_checkpoint or None, log)
        tb, ib = layout(W, H, True, True, st.captions)
        size = gen_size(st.image_model, ib[2] / ib[3])
        log(f"Generating {len(need_ai)} image(s) with '{st.image_model}' at {size[0]}x{size[1]} ...")
        use_ip = bool(refs) and st.image_model in ("sd15-lcm", "sdxl-lcm")
        if refs and not use_ip:
            log("  (reference images: this model has no IP-Adapter - only the colour palette is used)")
        try:
            if st.image_model != "doodle":
                gen.load(use_ip_adapter=use_ip)
            for k, i in enumerate(need_ai):
                _check(cancel)
                pos, neg = styles.build_prompts(scenes[i]["image_prompt"], st.style, st.style_extra, st.negative, meta.get("character", ""), palette)
                seed = st.seed if st.seed >= 0 else int(time.time()) % 100000
                key = cache_key(st.image_model, pos, neg, seed + i, size, st.image_steps, bool(refs), use_color, st.ref_strength)
                p = project / "images" / f"{i:03}_{key}.png"
                if p.exists():
                    imgs[i] = Image.open(p).convert("RGB")
                else:
                    log(f"  image {k + 1}/{len(need_ai)}: {scenes[i]['image_prompt'][:70]}")
                    ref = refs[i % len(refs)] if (refs and use_ip) else None
                    imgs[i] = gen.generate(pos, neg, seed + i, size, st.image_steps or None, ref, st.ref_strength, color=use_color)
                    imgs[i].save(p)
                progress(0.20 + 0.35 * (k + 1) / len(need_ai), f"Image {k + 1}/{len(need_ai)}")
        finally:
            gen.unload()
    progress(0.55, "Animating ...")

    # ---------------- C. animate scenes
    ink_rgb = lineart.hex_to_rgb(st.text_color)
    hand = hands.load_hand(st.hand, H, st.hand_scale or None, st.custom_hand, st.custom_hand_tip)
    segs, durations, offsets, srt, chap = [], [], [], [], []
    t_cursor = 0.0
    audio_parts = []
    for i, sc in enumerate(scenes):
        _check(cancel)
        text = (sc.get("on_screen_text") or "").strip()
        layers = _scene_layers(st, W, H, theme, use_color, text, imgs[i], font_path, None, ink_rgb)
        speech = len(wavs[i]) / SR
        seg_path = project / "segments" / f"{i:03}.mp4"
        cap_font = font_path

        def overlay_factory(lead, speech_sec, _txt=sc["narration"]):
            if not st.captions:
                return None
            return video.CaptionTrack(video.phrase_timings(_txt, lead, speech_sec), W, H, cap_font)

        n_frames, total, lead = _render_segment(st, W, H, theme, bg, layers, hand, seg_path, speech, fps,
                                                i == n - 1, overlay_factory, cancel, st.seed + i * 13)
        # audio padded to the exact segment length
        a = np.zeros(int(round(total * SR)), np.float32)
        s0 = int(lead * SR)
        a[s0:s0 + len(wavs[i])] = wavs[i][:len(a) - s0]
        audio_parts.append(a)
        for txt, ta, tb_ in video.phrase_timings(sc["narration"], t_cursor + lead, speech):
            srt.append((txt, ta, tb_))
        chap.append((t_cursor, text or short_label(sc["narration"], 5)))
        segs.append(seg_path); durations.append(total); t_cursor += total
        progress(0.55 + 0.38 * (i + 1) / n, f"Scene {i + 1}/{n} rendered")
        log(f"  scene {i + 1}/{n} done ({total:.1f}s)")

    # ---------------- D. assemble
    _check(cancel)
    progress(0.94, "Assembling final video ...")
    all_video = project / "video_only.mp4"
    video.concat_segments(segs, all_video)
    full_audio = np.concatenate(audio_parts)
    audio_path = project / "voice_track.wav"
    sf.write(str(audio_path), full_audio, SR)
    safe = re.sub(r"[^\w\- ]+", "", meta.get("title", "video")).strip().replace(" ", "_")[:50] or "video"
    final = project / f"{safe}.mp4"
    video.mux_final(all_video, audio_path, final, t_cursor, st.music, st.music_vol)

    # ---------------- E. YouTube extras
    thumb = project / "thumbnail.jpg"
    try:
        pick = max(range(n), key=lambda i: durations[i]) if n > 1 else 0
        lay = _scene_layers(st, W, H, theme, use_color, "", imgs[pick] if imgs[pick] is not None else None, font_path, None, ink_rgb)
        from .drawing import compose_final
        fr = compose_final(lay, bg, W, H) if lay else np.full((H, W, 3), 255, np.uint8)
        youtube.make_thumbnail(fr, meta.get("title", ""), font_path, thumb)
    except Exception as e:
        log(f"! thumbnail failed: {e}")
    srt_path = project / "subtitles.srt"
    video.write_srt(srt, srt_path)
    desc = project / "youtube.txt"
    ch = youtube.chapters(chap)
    desc.write_text(f"TITLE:\n{meta.get('title', '')}\n\nDESCRIPTION:\n{youtube.description_text(meta, ch)}\n\nTAGS:\n{', '.join(meta.get('tags', []))}\n", encoding="utf-8")
    progress(1.0, "Done")
    log(f"Finished in {time.time() - t_start:.0f}s -> {final}")
    res = dict(video=str(final), thumbnail=str(thumb), subtitles=str(srt_path), youtube=str(desc), project=str(project),
                duration=t_cursor)
    zip_path = project / f"{safe}_all_assets.zip"
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for k in ("video", "thumbnail", "subtitles", "youtube"):
                if k in res and Path(res[k]).exists():
                    p = Path(res[k])
                    z.write(p, arcname=p.name)
        res["zip"] = str(zip_path)
    except Exception as e:
        log(f"! ZIP packaging warning: {e}")
        res["zip"] = None
    return res


# ----------------------------------------------------------------------------- quick tool: images -> whiteboard video
def images_to_video(st: Settings, image_paths, seconds_per_image=6.0, audio_path=None, log=print, progress=lambda f, d="": None, cancel=None):
    """No AI needed: animate user images as hand-drawn whiteboard drawings."""
    project = new_project("images")
    W, H = canvas_size(st.aspect, st.resolution)
    fps = int(st.fps)
    style = styles.STYLES[st.style]
    theme = style["theme"]
    bg = lineart.THEMES[theme]["bg"]
    use_color = style["color"] if st.use_color == "style" else (st.use_color == "yes")
    hand = hands.load_hand(st.hand, H, st.hand_scale or None, st.custom_hand, st.custom_hand_tip)
    segs, audio, total = [], [], 0.0
    n = len(image_paths)
    for i, p in enumerate(image_paths):
        _check(cancel)
        img = Image.open(p if isinstance(p, str) else p.name).convert("RGB")
        layers = _scene_layers(st, W, H, theme, use_color, "", img, None, None, None)
        seg = project / "segments" / f"{i:03}.mp4"
        _, dur, _ = _render_segment(st, W, H, theme, bg, layers, hand, seg, seconds_per_image, fps, i == n - 1, None, cancel, st.seed + i)
        segs.append(seg); audio.append(np.zeros(int(round(dur * SR)), np.float32)); total += dur
        progress((i + 1) / n, f"Image {i + 1}/{n}")
        log(f"  image {i + 1}/{n} animated")
    vo = project / "video_only.mp4"
    video.concat_segments(segs, vo)
    wav = project / "track.wav"
    if audio_path:
        x, sr = sf.read(audio_path if isinstance(audio_path, str) else audio_path.name, dtype="float32")
        x = x.mean(1) if x.ndim > 1 else x
        x = tts.resample(x, sr, SR)
        full = np.zeros(int(round(total * SR)), np.float32)
        full[:min(len(full), len(x))] = x[:len(full)]
    else:
        full = np.concatenate(audio)
    sf.write(str(wav), full, SR)
    out = project / "whiteboard_video.mp4"
    video.mux_final(vo, wav, out, total, None, 0.1, normalize=bool(audio_path))
    return str(out)
