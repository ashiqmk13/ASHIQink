import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from . import SR, resample


class EspeakEngine:
    name = "espeak"
    native_speed = False

    def __init__(self, device="cpu", log=print):
        self.exe = shutil.which("espeak-ng") or shutil.which("espeak")
        self.log = log

    def available(self):
        if self.exe:
            return True, ""
        return False, "espeak-ng not installed (Linux: apt install espeak-ng | macOS: brew install espeak-ng | Windows: github.com/espeak-ng/espeak-ng/releases)"

    def synth(self, text, lang_info, voice=None, speed=1.0, ref_audio=None):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "o.wav"
            subprocess.run([self.exe, "-v", lang_info["espeak"], "-s", "155", "-p", "45", "-w", str(out), text],
                           check=True, capture_output=True)
            x, sr = sf.read(str(out), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        return resample(x, sr, SR)

    def close(self):
        pass
