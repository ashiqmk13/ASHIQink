"""ffmpeg helpers: segment writer, concat, audio tools, captions, SRT."""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

from . import fonts


def ffmpeg_exe() -> str:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError("ffmpeg not found. Install it (apt install ffmpeg / brew install ffmpeg) or `pip install imageio-ffmpeg`.") from e


def run_ff(args, what="ffmpeg"):
    r = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error"] + args, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{what} failed: {r.stderr[-600:]}")


class SegmentWriter:
    def __init__(self, path, W, H, fps, crf=20, preset="veryfast"):
        self.path = str(path)
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-g", str(int(fps) * 2), self.path]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        err = self.p.stderr.read().decode(errors="ignore")
        rc = self.p.wait()
        if rc:
            raise RuntimeError("ffmpeg encode failed: " + err[-500:])

    def abort(self):
        try:
            self.p.kill()
        except Exception:
            pass


def concat_segments(paths, out):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in paths:
            f.write(f"file '{Path(p).resolve().as_posix()}'\n")
        lst = f.name
    try:
        run_ff(["-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", str(out)], "concat")
    finally:
        Path(lst).unlink(missing_ok=True)


def change_tempo(x: np.ndarray, sr: int, speed: float) -> np.ndarray:
    if abs(speed - 1.0) < 0.02:
        return x
    speed = min(2.0, max(0.5, speed))
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a.wav", Path(td) / "b.wav"
        sf.write(str(a), x, sr)
        run_ff(["-i", str(a), "-filter:a", f"atempo={speed:.3f}", str(b)], "atempo")
        y, _ = sf.read(str(b), dtype="float32")
    return y


def mux_final(video, audio_wav, out, duration, music=None, music_vol=0.12, normalize=True):
    cmd = ["-i", str(video), "-i", str(audio_wav)]
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(music)]
        f = f"[2:a]volume={music_vol}[m];[1:a][m]amix=inputs=2:duration=first:dropout_transition=0[a0];"
    else:
        f = "[1:a]anull[a0];"
    f += "[a0]loudnorm=I=-16:TP=-1.5:LRA=11[a]" if normalize else "[a0]anull[a]"
    cmd += ["-filter_complex", f, "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-t", f"{duration:.3f}", "-movflags", "+faststart", str(out)]
    run_ff(cmd, "mux")


# ----------------------------------------------------------------------------- captions
def caption_phrases(text: str, max_chars=62):
    import re
    sents = [s for s in re.split(r"(?<=[.!?…।॥。！？])\s*", text) if s.strip()]
    out = []
    for s in sents:
        s = s.strip()
        if len(s) <= max_chars:
            out.append(s); continue
        cur = ""
        sep = "" if " " not in s else " "
        for w in (list(s) if not sep else s.split(" ")):
            if cur and len(cur) + len(w) + 1 > max_chars:
                out.append(cur); cur = w
            else:
                cur = (cur + sep + w) if cur else w
        if cur:
            out.append(cur)
    return out or [text]


def phrase_timings(text, t0, dur):
    ph = caption_phrases(text)
    w = np.array([max(1, len(p)) for p in ph], float)
    edges = t0 + np.concatenate([[0], np.cumsum(w) / w.sum()]) * dur
    return [(ph[i], float(edges[i]), float(edges[i + 1])) for i in range(len(ph))]


class CaptionTrack:
    def __init__(self, items, W, H, font_path):
        self.items, self.W, self.H = items, W, H
        self.cache = {}
        self.font = fonts.load_font(font_path, max(18, int(H * 0.045)))
        self.font_path = font_path

    def _render(self, text):
        W, H = self.W, self.H
        f = self.font
        lines = fonts.wrap_text(text, f, W * 0.84, any(ord(c) > 0x2E80 for c in text))[:3]
        asc, desc = f.getmetrics()
        lh = int((asc + desc) * 1.2)
        bh = lh * len(lines) + int(H * 0.03)
        im = Image.new("RGBA", (W, bh), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        tw = max(f.getlength(l) for l in lines)
        pad = int(H * 0.018)
        x0 = (W - tw) / 2 - pad
        d.rounded_rectangle([x0, 2, x0 + tw + 2 * pad, bh - 2], radius=int(H * 0.012), fill=(20, 20, 24, 170))
        y = int(H * 0.012)
        for l in lines:
            d.text(((W - f.getlength(l)) / 2, y), l, font=f, fill=(255, 255, 255, 255))
            y += lh
        arr = np.asarray(im, np.float32)
        a = arr[..., 3:4] / 255.0
        return arr[..., :3] * a, 1.0 - a, bh

    def __call__(self, frame, t):
        for text, a, b in self.items:
            if a <= t < b:
                if text not in self.cache:
                    self.cache[text] = self._render(text)
                pm, inv, bh = self.cache[text]
                y0 = self.H - bh - int(self.H * 0.035)
                roi = frame[y0:y0 + bh]
                roi[:] = (roi * inv + pm).astype(np.uint8)
                break
        return frame


def fmt_srt(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(entries, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, (txt, a, b) in enumerate(entries, 1):
            f.write(f"{i}\n{fmt_srt(a)} --> {fmt_srt(b)}\n{txt}\n\n")
