"""Image generation. Everything runs locally; weights are downloaded from the Hugging Face hub on first use
(into ./data/huggingface). 'doodle' needs no download at all."""
import hashlib
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

from . import hardware

IMAGE_MODELS = {
    "doodle": dict(
        label="Instant doodles - no AI, no download (simple icon placeholders)",
        size="0 MB", license="n/a"),
    "sd15-lcm": dict(
        label="Stable Diffusion 1.5 + LCM - best for CPU (4-6 steps, ~4 GB)",
        repo="stable-diffusion-v1-5/stable-diffusion-v1-5", lora="latent-consistency/lcm-lora-sdv1-5",
        steps=5, cfg=1.8, base=512, size="~4.5 GB", license="CreativeML OpenRAIL-M (commercial use allowed)", sdxl=False),
    "sdxl-lcm": dict(
        label="SDXL 1.0 + LCM - high quality, GPU recommended (~7 GB)",
        repo="stabilityai/stable-diffusion-xl-base-1.0", lora="latent-consistency/lcm-lora-sdxl",
        steps=6, cfg=1.8, base=1024, size="~7.5 GB", license="CreativeML OpenRAIL++-M (commercial use allowed)", sdxl=True),
    "flux-schnell": dict(
        label="FLUX.1-schnell - best quality, needs strong GPU (>=16 GB VRAM) (~24 GB)",
        repo="black-forest-labs/FLUX.1-schnell", steps=4, cfg=0.0, base=1024, size="~24 GB",
        license="Apache-2.0", sdxl=False, flux=True),
}
IMAGE_CHOICES = [(v["label"], k) for k, v in IMAGE_MODELS.items()]


def gen_size(model_key: str, aspect_ratio: float):
    """(w,h), multiples of 8, near the model's native pixel count, for aspect ratio w/h."""
    base = IMAGE_MODELS.get(model_key, {}).get("base", 512)
    long_side = 640 if base == 512 else base
    if aspect_ratio > 1.2:
        w, h = long_side, long_side / aspect_ratio
    elif aspect_ratio < 0.83:
        h, w = long_side, long_side * aspect_ratio
    else:
        w = h = base
    return int(round(w / 8) * 8), int(round(h / 8) * 8)


# ----------------------------------------------------------------------------- placeholder doodles
def _doodle(prompt: str, w: int, h: int, seed: int, color=True) -> Image.Image:
    S = 2
    img = Image.new("RGB", (w * S, h * S), "white")
    d = ImageDraw.Draw(img)
    rnd = random.Random(seed)
    pal = [(255, 205, 60), (110, 190, 255), (240, 100, 90), (120, 205, 120), (190, 140, 230), (255, 160, 90)]
    lw = max(4, int(min(w, h) * 0.012)) * S
    K = (20, 20, 25)

    def col():
        return rnd.choice(pal) if color else None

    def oval(x, y, rx, ry, f=None):
        d.ellipse([x - rx, y - ry, x + rx, y + ry], fill=f, outline=K, width=lw)

    def sun(x, y, s):
        for i in range(10):
            a = i * math.pi / 5
            d.line([x + math.cos(a) * s * 1.25, y + math.sin(a) * s * 1.25, x + math.cos(a) * s * 1.7, y + math.sin(a) * s * 1.7], fill=K, width=lw)
        oval(x, y, s, s, (255, 210, 60) if color else None)

    def bulb(x, y, s):
        oval(x, y - s * 0.3, s * 0.8, s * 0.9, (255, 235, 120) if color else None)
        d.rectangle([x - s * 0.35, y + s * 0.5, x + s * 0.35, y + s * 1.0], fill=(200, 200, 205) if color else None, outline=K, width=lw)
        for i in range(-1, 2):
            a = -math.pi / 2 + i * 0.7
            d.line([x + math.cos(a) * s * 1.2, y - s * 0.3 + math.sin(a) * s * 1.3, x + math.cos(a) * s * 1.6, y - s * 0.3 + math.sin(a) * s * 1.75], fill=K, width=lw)

    def house(x, y, s):
        d.rectangle([x - s, y - s * 0.3, x + s, y + s], fill=col(), outline=K, width=lw)
        d.polygon([(x - s * 1.25, y - s * 0.3), (x, y - s * 1.3), (x + s * 1.25, y - s * 0.3)], fill=(235, 100, 90) if color else None, outline=K, width=lw)
        d.rectangle([x - s * 0.2, y + s * 0.2, x + s * 0.3, y + s], fill=(150, 100, 60) if color else None, outline=K, width=lw)

    def tree(x, y, s):
        d.rectangle([x - s * 0.18, y, x + s * 0.18, y + s * 1.2], fill=(150, 100, 60) if color else None, outline=K, width=lw)
        oval(x, y - s * 0.3, s * 0.9, s * 0.9, (110, 195, 110) if color else None)

    def cloud(x, y, s):
        for dx, dy, r in ((-0.7, 0.1, 0.6), (0, -0.25, 0.8), (0.75, 0.1, 0.6)):
            oval(x + dx * s, y + dy * s, r * s, r * s, (225, 240, 255) if color else None)
        d.rectangle([x - 0.7 * s, y + 0.1 * s, x + 0.75 * s, y + 0.62 * s], fill=(225, 240, 255) if color else "white")
        d.line([x - 0.7 * s, y + 0.7 * s, x + 0.75 * s, y + 0.7 * s], fill=K, width=lw)

    def person(x, y, s):
        oval(x, y - s * 0.9, s * 0.35, s * 0.35, (255, 220, 185) if color else None)
        d.line([x, y - s * 0.55, x, y + s * 0.4], fill=K, width=lw)
        d.line([x - s * 0.5, y - s * 0.2, x + s * 0.5, y - s * 0.2], fill=K, width=lw)
        d.line([x, y + s * 0.4, x - s * 0.4, y + s * 1.1], fill=K, width=lw)
        d.line([x, y + s * 0.4, x + s * 0.4, y + s * 1.1], fill=K, width=lw)

    def chart(x, y, s):
        d.line([x - s, y - s, x - s, y + s, x + s, y + s], fill=K, width=lw)
        for i, hh in enumerate((0.5, 0.9, 1.4, 1.9)):
            x0 = x - s * 0.8 + i * s * 0.45
            d.rectangle([x0, y + s - hh * s * 0.8, x0 + s * 0.32, y + s], fill=pal[i % 6] if color else None, outline=K, width=lw)
        d.line([x - s * 0.8, y + s * 0.4, x - s * 0.1, y - s * 0.1, x + s * 0.5, y - s * 0.3, x + s * 0.9, y - s * 0.9], fill=K, width=lw)

    def clock(x, y, s):
        oval(x, y, s, s, (255, 255, 240) if color else None)
        d.line([x, y, x, y - s * 0.7], fill=K, width=lw); d.line([x, y, x + s * 0.5, y + s * 0.2], fill=K, width=lw)

    def moon(x, y, s):
        oval(x, y, s, s, (255, 235, 140) if color else None)
        oval(x + s * 0.45, y - s * 0.15, s * 0.85, s * 0.85, "white")
        for i in range(3):
            sx, sy = x + s * (1.4 + i * 0.5), y - s * (0.6 - 0.5 * (i % 2))
            d.polygon([(sx, sy - s * .25), (sx + s * .08, sy - s * .08), (sx + s * .25, sy - s * .05), (sx + s * .1, sy + s * .06), (sx + s * .15, sy + s * .25), (sx, sy + s * .12), (sx - s * .15, sy + s * .25), (sx - s * .1, sy + s * .06), (sx - s * .25, sy - s * .05), (sx - s * .08, sy - s * .08)], outline=K, fill=(255, 220, 80) if color else None)

    def heart(x, y, s):
        d.polygon([(x, y + s), (x - s * 1.1, y - s * 0.1), (x - s * 0.6, y - s * 0.8), (x, y - s * 0.4), (x + s * 0.6, y - s * 0.8), (x + s * 1.1, y - s * 0.1)], fill=(235, 90, 110) if color else None, outline=K, width=lw)

    def book(x, y, s):
        d.polygon([(x - s * 1.1, y - s * 0.6), (x, y - s * 0.4), (x, y + s * 0.7), (x - s * 1.1, y + s * 0.5)], fill=(120, 190, 255) if color else None, outline=K, width=lw)
        d.polygon([(x + s * 1.1, y - s * 0.6), (x, y - s * 0.4), (x, y + s * 0.7), (x + s * 1.1, y + s * 0.5)], fill=(150, 215, 255) if color else None, outline=K, width=lw)

    icons = dict(sun=sun, bulb=bulb, house=house, tree=tree, cloud=cloud, person=person, chart=chart, clock=clock, moon=moon, heart=heart, book=book)
    kw = {"sleep": "moon", "night": "moon", "moon": "moon", "dream": "moon", "idea": "bulb", "think": "bulb", "brain": "bulb", "learn": "book",
          "book": "book", "study": "book", "school": "book", "money": "chart", "growth": "chart", "profit": "chart", "business": "chart",
          "data": "chart", "market": "chart", "people": "person", "team": "person", "person": "person", "human": "person", "teacher": "person",
          "home": "house", "house": "house", "building": "house", "plant": "tree", "tree": "tree", "nature": "tree", "forest": "tree",
          "time": "clock", "clock": "clock", "hour": "clock", "health": "heart", "heart": "heart", "love": "heart", "sun": "sun", "energy": "sun",
          "weather": "cloud", "cloud": "cloud", "sky": "cloud", "internet": "cloud"}
    low = prompt.lower()
    chosen = []
    for k, v in kw.items():
        if k in low and v not in chosen:
            chosen.append(v)
    names = list(icons)
    rnd.shuffle(names)
    for n in names:
        if len(chosen) >= 3:
            break
        if n not in chosen:
            chosen.append(n)
    chosen = chosen[:3]
    n = len(chosen)
    unit = min(w * S / (n * 2.9), h * S / 3.2)
    for i, name in enumerate(chosen):
        cx = w * S * (i + 0.5) / n
        cy = h * S * 0.5 + rnd.uniform(-0.04, 0.04) * h * S
        icons[name](cx, cy, unit)
    d.line([w * S * 0.08, h * S * 0.93, w * S * 0.92, h * S * 0.93 + rnd.uniform(-6, 6)], fill=K, width=lw)
    return img.resize((w, h), Image.LANCZOS)


# ----------------------------------------------------------------------------- diffusers wrapper
class ImageGenerator:
    def __init__(self, model_key="sd15-lcm", device="auto", custom_checkpoint=None, log=print):
        self.key = model_key
        self.spec = IMAGE_MODELS[model_key]
        self.device = hardware.resolve_device(device)
        self.custom = custom_checkpoint if custom_checkpoint and Path(custom_checkpoint).exists() else None
        self.log = log
        self.pipe = None
        self.ip_loaded = False

    # -- loading
    def load(self, use_ip_adapter=False):
        if self.key == "doodle" or self.pipe is not None:
            return
        import torch
        from diffusers import LCMScheduler
        spec, dev = self.spec, self.device
        cuda = dev == "cuda"
        dtype = torch.float16 if cuda else torch.float32
        self.log(f"Loading image model '{self.key}' on {dev} (first run downloads {spec['size']}) ...")
        if spec.get("flux"):
            from diffusers import FluxPipeline
            pipe = FluxPipeline.from_pretrained(spec["repo"], torch_dtype=torch.bfloat16)
            if cuda:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cpu")
            self.pipe = pipe
            return
        if spec.get("sdxl"):
            from diffusers import StableDiffusionXLPipeline as P
            kw = dict(torch_dtype=dtype, use_safetensors=True)
        else:
            from diffusers import StableDiffusionPipeline as P
            kw = dict(torch_dtype=dtype, use_safetensors=True, safety_checker=None, requires_safety_checker=False)
        if self.custom:
            pipe = P.from_single_file(self.custom, torch_dtype=dtype, **({} if spec.get("sdxl") else dict(safety_checker=None)))
        else:
            try:
                pipe = P.from_pretrained(spec["repo"], variant="fp16" if cuda else None, **kw)
            except Exception as e:
                self.log(f"  (variant load failed: {str(e)[:80]} - retrying without variant)")
                pipe = P.from_pretrained(spec["repo"], **kw)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        try:
            pipe.load_lora_weights(spec["lora"])
            pipe.fuse_lora()
        except Exception as e:
            self.log(f"! LCM LoRA failed to load ({e}). Falling back to plain sampler with more steps.")
            self.spec = dict(spec, steps=20, cfg=6.0)
        if use_ip_adapter:
            try:
                if spec.get("sdxl"):
                    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
                else:
                    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
                self.ip_loaded = True
            except Exception as e:
                self.log(f"! IP-Adapter (reference style) unavailable: {str(e)[:120]}")
        vram = hardware.detect()["vram_gb"]
        if cuda and vram and vram < 10 and spec.get("sdxl"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(dev)
        if not cuda:
            pipe.enable_attention_slicing()
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
        self.pipe = pipe

    def unload(self):
        self.pipe = None
        hardware.free_memory()

    # -- generation
    def generate(self, prompt, negative, seed, size, steps=None, ref=None, ref_scale=0.6, color=True) -> Image.Image:
        w, h = size
        if self.key == "doodle":
            return _doodle(prompt, w, h, seed, color=color)
        import torch
        self.load()
        g = torch.Generator("cpu").manual_seed(int(seed) & 0x7FFFFFFF)
        kw = dict(prompt=prompt, num_inference_steps=int(steps or self.spec["steps"]),
                  guidance_scale=self.spec["cfg"], width=w, height=h, generator=g)
        if self.spec.get("flux"):
            kw["max_sequence_length"] = 256
        else:
            kw["negative_prompt"] = negative
        if self.ip_loaded and ref is not None:
            self.pipe.set_ip_adapter_scale(float(ref_scale))
            kw["ip_adapter_image"] = ref
        elif self.ip_loaded:
            self.pipe.set_ip_adapter_scale(0.0)
            kw["ip_adapter_image"] = Image.new("RGB", (224, 224), "white")
        with torch.inference_mode():
            return self.pipe(**kw).images[0]


def cache_key(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]
