import numpy as np
from . import SR, GAP, resample

KOKORO_VOICES = {
    "a": ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_michael", "am_fenrir", "am_puck"],
    "b": ["bf_emma", "bf_isabella", "bm_george", "bm_lewis", "bm_daniel"],
    "e": ["ef_dora", "em_alex", "em_santa"],
    "f": ["ff_siwis"],
    "h": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "i": ["if_sara", "im_nicola"],
    "j": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "p": ["pf_dora", "pm_alex", "pm_santa"],
    "z": ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"],
}


class KokoroEngine:
    name = "kokoro"
    native_speed = True

    def __init__(self, device="cpu", log=print):
        self.device = "cuda" if device == "cuda" else "cpu"
        self.log = log
        self.pipes = {}

    def available(self):
        try:
            import kokoro  # noqa: F401
            return True, ""
        except Exception as e:
            return False, f"kokoro package missing ({e}). Run: pip install kokoro soundfile"

    def _pipe(self, code):
        if code not in self.pipes:
            from kokoro import KPipeline
            self.log(f"Loading Kokoro ({code}) - first run downloads ~330 MB ...")
            self.pipes[code] = KPipeline(lang_code=code, device=self.device)
        return self.pipes[code]

    def synth(self, text, lang_info, voice=None, speed=1.0, ref_audio=None):
        code = lang_info["kokoro"]
        if not voice or voice == "auto" or voice not in KOKORO_VOICES[code]:
            voice = KOKORO_VOICES[code][0]
        pipe = self._pipe(code)
        parts = []
        for _, _, audio in pipe(text, voice=voice, speed=float(speed), split_pattern=r"\n+"):
            if audio is None:
                continue
            a = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            parts.append(a.astype(np.float32).ravel())
            parts.append(np.zeros(int(GAP * 24000), np.float32))
        if not parts:
            raise RuntimeError("Kokoro returned no audio")
        return resample(np.concatenate(parts), 24000, SR)

    def close(self):
        self.pipes.clear()
