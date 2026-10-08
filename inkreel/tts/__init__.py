"""Text-to-speech manager. Engines: Kokoro, Chatterbox, MMS-TTS, espeak-ng (offline fallback)."""
import re
import numpy as np

from ..langs import LANGS

SR = 24000
GAP = 0.18          # seconds of silence between sentence chunks

ENGINE_LABELS = {
    "auto": "Auto (best available for the language)",
    "kokoro": "Kokoro 82M - fast, great on CPU (9 languages, Apache-2.0)",
    "chatterbox": "Chatterbox - natural + voice cloning (23 languages, MIT; GPU preferred)",
    "mms": "MMS-TTS - 1000+ languages incl. Malayalam/Tamil/Telugu (CC-BY-NC: non-commercial!)",
    "espeak": "espeak-ng - robotic but offline & 100+ languages (zero download)",
}
ENGINE_CHOICES = [(v, k) for k, v in ENGINE_LABELS.items()]


def chunk_text(text: str, max_chars=220):
    sents = [s.strip() for s in re.split(r"(?<=[.!?…।॥])\s+|(?<=[。！？])|\n+", text) if s and s.strip()]
    out, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > max_chars:
            out.append(cur); cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    final = []
    for c in out:                      # hard-split extremely long sentences
        while len(c) > max_chars * 1.6:
            cut = c.rfind(",", 0, max_chars) or c.rfind(" ", 0, max_chars)
            cut = cut if cut > 20 else max_chars
            final.append(c[:cut].strip()); c = c[cut:].lstrip(" ,")
        final.append(c)
    return final or [text]


def resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def trim_and_normalize(x: np.ndarray, sr=SR, thr=0.006, keep=0.08, peak=0.92) -> np.ndarray:
    x = np.asarray(x, np.float32).ravel()
    if x.size == 0:
        return x
    idx = np.nonzero(np.abs(x) > thr)[0]
    if idx.size:
        a, b = max(0, idx[0] - int(keep * sr)), min(len(x), idx[-1] + int(keep * sr))
        x = x[a:b]
    m = float(np.abs(x).max()) or 1.0
    return x * (peak / m) if m > 0.01 else x


def engine_support(lang: str):
    L = LANGS.get(lang, {})
    return dict(kokoro=bool(L.get("kokoro")), chatterbox=bool(L.get("chatterbox")),
                mms=bool(L.get("mms")), espeak=bool(L.get("espeak")))


def get_engine(name: str, device: str, log=print):
    if name == "kokoro":
        from .kokoro_engine import KokoroEngine; return KokoroEngine(device, log)
    if name == "chatterbox":
        from .chatterbox_engine import ChatterboxEngine; return ChatterboxEngine(device, log)
    if name == "mms":
        from .mms_engine import MMSEngine; return MMSEngine(device, log)
    if name == "espeak":
        from .espeak_engine import EspeakEngine; return EspeakEngine(device, log)
    raise ValueError(name)


def pick_engine(preferred: str, lang: str, device: str, log=print):
    """Return an engine instance that really works for `lang`, trying fallbacks in order."""
    order = [preferred] if preferred != "auto" else []
    order += [e for e in ("kokoro", "chatterbox", "mms", "espeak") if e not in order]
    sup = engine_support(lang)
    errors = []
    for name in order:
        if not sup.get(name):
            if name == preferred:
                log(f"! {name} does not support '{lang}' - looking for an alternative")
            continue
        try:
            eng = get_engine(name, device, log)
            ok, why = eng.available()
            if ok:
                if preferred not in ("auto", name):
                    log(f"-> using {name} instead of {preferred}")
                return eng
            errors.append(f"{name}: {why}")
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError(f"No TTS engine available for '{lang}'.\n" + "\n".join(errors))


def voices_for(engine: str, lang: str):
    if engine in ("kokoro", "auto") and LANGS.get(lang, {}).get("kokoro"):
        from .kokoro_engine import KOKORO_VOICES
        return KOKORO_VOICES[LANGS[lang]["kokoro"]]
    return []
