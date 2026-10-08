import gc
import os


def _ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return None


def detect() -> dict:
    info = dict(device="cpu", gpu_name=None, vram_gb=0.0, cpu_threads=os.cpu_count() or 4,
                ram_gb=_ram_gb(), torch=None)
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(device="cuda", gpu_name=p.name, vram_gb=round(p.total_memory / 1e9, 1))
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info.update(device="mps", gpu_name="Apple GPU")
    except Exception:
        pass
    return info


def resolve_device(pref: str = "auto") -> str:
    hw = detect()
    if pref in (None, "", "auto"):
        return hw["device"]
    if pref == "cpu":
        return "cpu"
    return pref if hw["device"] == pref else "cpu"


def describe() -> str:
    hw = detect()
    ram = f"{hw['ram_gb']:.0f} GB RAM" if hw["ram_gb"] else "RAM unknown"
    if hw["device"] == "cuda":
        g = f"GPU: {hw['gpu_name']} ({hw['vram_gb']} GB VRAM)"
    elif hw["device"] == "mps":
        g = "GPU: Apple Silicon (MPS)"
    else:
        g = "No GPU detected -> running on CPU (slower, fully supported)"
    return f"{g}  |  CPU threads: {hw['cpu_threads']}  |  {ram}  |  torch {hw['torch'] or 'not installed'}"


def recommend() -> dict:
    hw = detect()
    if hw["device"] == "cuda" and hw["vram_gb"] >= 14:
        return dict(image_model="sdxl-lcm", llm="Qwen2.5-7B (best quality)", note="Strong GPU: SDXL + 7B LLM are fine.")
    if hw["device"] == "cuda":
        return dict(image_model="sd15-lcm", llm="Qwen2.5-1.5B (default)", note="Small GPU: SD1.5-LCM is fast and safe.")
    return dict(image_model="sd15-lcm", llm="Qwen2.5-1.5B (default)", note="CPU: SD1.5-LCM (5 steps) ~ 15-60 s per image.")


def free_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
