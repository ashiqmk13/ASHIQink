import numpy as np
from . import SR, GAP, resample, chunk_text


class MMSEngine:
    """Meta MMS-TTS (VITS) through transformers. NOTE: weights are CC-BY-NC 4.0 (non-commercial)."""
    name = "mms"
    native_speed = False

    def __init__(self, device="cpu", log=print):
        self.device = "cuda" if device == "cuda" else "cpu"
        self.log = log
        self.models = {}

    def available(self):
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True, ""
        except Exception as e:
            return False, str(e)

    def _load(self, code):
        if code not in self.models:
            from transformers import VitsModel, AutoTokenizer
            repo = f"facebook/mms-tts-{code}"
            self.log(f"Loading {repo} (first run downloads ~150 MB) ...")
            tok = AutoTokenizer.from_pretrained(repo)
            if getattr(tok, "is_uroman", False):
                raise RuntimeError(f"{repo} needs the 'uroman' romanizer which is not bundled; pick another engine for this language.")
            model = VitsModel.from_pretrained(repo).to(self.device).eval()
            self.models[code] = (tok, model)
        return self.models[code]

    def synth(self, text, lang_info, voice=None, speed=1.0, ref_audio=None):
        import torch
        tok, model = self._load(lang_info["mms"])
        parts = []
        for chunk in chunk_text(text, 180):
            inp = tok(chunk, return_tensors="pt").to(self.device)
            with torch.no_grad():
                wav = model(**inp).waveform[0].cpu().numpy()
            parts.append(resample(wav, model.config.sampling_rate, SR))
            parts.append(np.zeros(int(GAP * SR), np.float32))
        return np.concatenate(parts)

    def close(self):
        self.models.clear()
