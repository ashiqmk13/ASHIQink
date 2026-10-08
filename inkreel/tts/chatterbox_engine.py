"""Chatterbox TTS (Resemble AI, MIT). It pins old torch/transformers versions, so by default it runs in its own
virtualenv (.venv_chatterbox, created by `python install.py --with-chatterbox`) as a persistent worker process.
If `chatterbox` happens to be importable in the main environment it is used directly instead."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

from . import SR, GAP, resample, chunk_text
from ..config import CHATTERBOX_VENV

WORKER = Path(__file__).with_name("chatterbox_worker.py")


def _venv_python():
    for p in (CHATTERBOX_VENV / "bin" / "python", CHATTERBOX_VENV / "Scripts" / "python.exe"):
        if p.exists():
            return str(p)
    return None


class ChatterboxEngine:
    name = "chatterbox"
    native_speed = False

    def __init__(self, device="cpu", log=print):
        self.device = device if device in ("cuda", "mps") else "cpu"
        self.log = log
        self.proc = None
        self.direct = None

    def available(self):
        try:
            import chatterbox  # noqa: F401
            self.mode = "direct"
            return True, ""
        except Exception:
            pass
        if _venv_python():
            self.mode = "worker"
            return True, ""
        return False, "Chatterbox not installed. Run: python install.py --with-chatterbox"

    # ---- worker process (isolated venv)
    def _start(self):
        if self.proc and self.proc.poll() is None:
            return
        self.log("Starting Chatterbox worker (first run downloads ~3 GB) ...")
        self.proc = subprocess.Popen([_venv_python(), str(WORKER), self.device], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1)
        ready = self.proc.stdout.readline()
        if '"ready"' not in ready:
            raise RuntimeError("Chatterbox worker failed to start (see console output)")

    def _worker_job(self, text, lang, out, ref):
        self._start()
        self.proc.stdin.write(json.dumps(dict(text=text, lang=lang, out=out, ref=ref)) + "\n")
        self.proc.stdin.flush()
        resp = json.loads(self.proc.stdout.readline() or "{}")
        if not resp.get("ok"):
            raise RuntimeError("Chatterbox: " + str(resp.get("error", "worker died")))

    def synth(self, text, lang_info, voice=None, speed=1.0, ref_audio=None):
        lang = lang_info["chatterbox"]
        parts = []
        with tempfile.TemporaryDirectory() as td:
            for i, chunk in enumerate(chunk_text(text, 240)):
                out = str(Path(td) / f"c{i}.wav")
                if self.mode == "direct":
                    self._direct(chunk, lang, out, ref_audio)
                else:
                    self._worker_job(chunk, lang, out, ref_audio)
                x, sr = sf.read(out, dtype="float32")
                x = x.mean(1) if x.ndim > 1 else x
                parts.append(resample(x, sr, SR)); parts.append(np.zeros(int(GAP * SR), np.float32))
        return np.concatenate(parts)

    def _direct(self, text, lang, out, ref):
        import torch
        if self.direct is None:
            if self.device == "cpu":
                _orig = torch.load
                torch.load = lambda *a, **k: _orig(*a, **{**k, "map_location": torch.device("cpu")})
            try:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS as M
            except Exception:
                from chatterbox.tts import ChatterboxTTS as M
            self.direct = M.from_pretrained(device=self.device)
        kw = dict(audio_prompt_path=ref) if ref else {}
        if "Multilingual" in type(self.direct).__name__:
            kw["language_id"] = lang
        wav = self.direct.generate(text, **kw)
        sf.write(out, wav.squeeze().cpu().numpy(), self.direct.sr)

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close(); self.proc.terminate()
            except Exception:
                pass
        self.proc = None
        self.direct = None
