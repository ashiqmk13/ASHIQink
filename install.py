"""One-command installer.

  python install.py                 # auto-detect GPU, create .venv, install, download CPU-friendly models
  python install.py --cpu           # force CPU build of torch
  python install.py --gpu           # force CUDA build of torch (cu124)
  python install.py --profile gpu   # download SDXL + 7B LLM too
  python install.py --with-chatterbox --extra-langs   # optional extras
  python install.py --no-models     # skip model downloads (they happen on first use anyway)
  python install.py --here          # install into the current Python instead of creating .venv (Colab)
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd, **kw):
    print("\n$", " ".join(map(str, cmd)))
    subprocess.check_call(list(map(str, cmd)), **kw)


def has_nvidia():
    return shutil.which("nvidia-smi") is not None


def make_venv(path: Path):
    if not path.exists():
        print(f"Creating virtualenv {path} ...")
        venv.EnvBuilder(with_pip=True).create(path)
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def torch_args(gpu: bool):
    if gpu:
        return ["torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu124"]
    if platform.system() == "Darwin":
        return ["torch", "torchaudio"]
    return ["torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--cpu", action="store_true"); g.add_argument("--gpu", action="store_true")
    ap.add_argument("--profile", choices=["minimal", "cpu", "gpu"], default=None)
    ap.add_argument("--no-models", action="store_true")
    ap.add_argument("--with-chatterbox", action="store_true")
    ap.add_argument("--extra-langs", action="store_true", help="Japanese + Chinese support for Kokoro")
    ap.add_argument("--with-llamacpp", action="store_true", help="faster CPU LLM via llama-cpp-python wheels")
    ap.add_argument("--here", action="store_true")
    a = ap.parse_args()

    if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
        print("! Python 3.10 - 3.12 is recommended (kokoro / torch wheels). You have", sys.version.split()[0])
    gpu = a.gpu or (not a.cpu and has_nvidia())
    print("GPU build:" , gpu)
    py = sys.executable if a.here else str(make_venv(ROOT / ".venv"))

    run([py, "-m", "pip", "install", "--upgrade", "pip", "wheel"])
    already = subprocess.call([py, "-c", "import torch"], stderr=subprocess.DEVNULL) == 0
    if not already:
        run([py, "-m", "pip", "install"] + torch_args(gpu))
    else:
        print("torch already present - keeping it")
    run([py, "-m", "pip", "install", "-r", ROOT / "requirements.txt"])
    if a.extra_langs:
        run([py, "-m", "pip", "install", "misaki[ja]", "misaki[zh]"])
    if a.with_llamacpp:
        idx = "https://abetlen.github.io/llama-cpp-python/whl/" + ("cu124" if gpu else "cpu")
        run([py, "-m", "pip", "install", "llama-cpp-python", "--extra-index-url", idx])

    if a.with_chatterbox:
        cpy = make_venv(ROOT / ".venv_chatterbox")
        run([cpy, "-m", "pip", "install", "--upgrade", "pip", "wheel"])
        if platform.system() != "Darwin" and not gpu:
            run([cpy, "-m", "pip", "install", "torch==2.6.0", "torchaudio==2.6.0", "--index-url", "https://download.pytorch.org/whl/cpu"])
        run([cpy, "-m", "pip", "install", "chatterbox-tts", "soundfile"])

    # system helpers (best effort, never fatal)
    if platform.system() == "Linux" and shutil.which("apt-get"):
        sudo = [] if os.geteuid() == 0 else ["sudo"]
        try:
            run(sudo + ["apt-get", "install", "-y", "espeak-ng", "libraqm0", "ffmpeg"])
        except Exception:
            print("! could not apt-get install espeak-ng / libraqm0 / ffmpeg (optional; ffmpeg also comes via imageio-ffmpeg)")
    elif platform.system() == "Darwin" and shutil.which("brew"):
        subprocess.call(["brew", "install", "espeak-ng", "libraqm"])
    else:
        print("Optional: install espeak-ng (offline voices for 100+ languages) from https://github.com/espeak-ng/espeak-ng/releases")

    if not a.no_models:
        prof = a.profile or ("gpu" if gpu else "cpu")
        run([py, "-m", "inkreel.prefetch", "--profile", prof], cwd=ROOT)

    print("\nInstalled.  Start with:   " + ("run.bat" if os.name == "nt" else "./run.sh"))


if __name__ == "__main__":
    main()
