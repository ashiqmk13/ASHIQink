"""Local LLM for script analysis (scene prompts, on-screen text, title/description/tags, translation).
Backends: llama-cpp-python (GGUF, fastest on CPU) if installed, otherwise transformers. No API, no cloud.
Without any LLM a rule-based fallback still works (English keywords)."""
import re
from collections import Counter

from . import hardware
from .langs import LANGS
from .script_utils import keywords, short_label, split_sentences

LLM_MODELS = {
    "Qwen2.5-0.5B (tiny, fastest)": dict(hf="Qwen/Qwen2.5-0.5B-Instruct", gguf_repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                                         gguf_file="qwen2.5-0.5b-instruct-q4_k_m.gguf", size="~0.4 GB", small=True),
    "Qwen2.5-1.5B (default)": dict(hf="Qwen/Qwen2.5-1.5B-Instruct", gguf_repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                                   gguf_file="qwen2.5-1.5b-instruct-q4_k_m.gguf", size="~1.1 GB", small=True),
    "Qwen2.5-7B (best quality)": dict(hf="Qwen/Qwen2.5-7B-Instruct", gguf_repo="bartowski/Qwen2.5-7B-Instruct-GGUF",
                                      gguf_file="Qwen2.5-7B-Instruct-Q4_K_M.gguf", size="~4.7 GB", small=False),
    "None (rule-based, no AI)": None,
}
LLM_CHOICES = list(LLM_MODELS)


class LLM:
    def __init__(self, model_name, device="auto", log=print):
        self.spec = LLM_MODELS[model_name]
        self.device = hardware.resolve_device(device)
        self.log = log
        self.backend = None
        self.model = self.tok = None

    def load(self):
        if self.backend:
            return
        s = self.spec
        try:
            from llama_cpp import Llama
            self.log(f"Loading {s['gguf_file']} with llama.cpp (first run downloads {s['size']}) ...")
            self.model = Llama.from_pretrained(
                repo_id=s["gguf_repo"], filename=s["gguf_file"], n_ctx=4096, verbose=False,
                n_gpu_layers=-1 if self.device == "cuda" else 0, n_threads=hardware.detect()["cpu_threads"])
            self.backend = "llama"
            return
        except ImportError:
            pass
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.log(f"Loading {s['hf']} with transformers on {self.device} (first run downloads {s['size']} x2-3) ...")
        dtype = torch.float16 if self.device == "cuda" else (torch.float32 if s["small"] else torch.bfloat16)
        self.tok = AutoTokenizer.from_pretrained(s["hf"])
        self.model = AutoModelForCausalLM.from_pretrained(s["hf"], torch_dtype=dtype).to(self.device).eval()
        self.backend = "hf"

    def chat(self, system, user, max_new_tokens=160, temperature=0.4):
        self.load()
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.backend == "llama":
            r = self.model.create_chat_completion(messages=msgs, max_tokens=max_new_tokens, temperature=temperature)
            return r["choices"][0]["message"]["content"].strip()
        import torch
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                      temperature=max(temperature, 1e-3), top_p=0.9, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    def unload(self):
        self.model = self.tok = self.backend = None
        hardware.free_memory()


def _field(text, name):
    m = re.search(rf"^\s*\**{name}\**\s*[:：]\s*(.+)$", text, re.I | re.M)
    return m.group(1).strip().strip('"*') if m else ""


# ----------------------------------------------------------------------------- analysis
def fallback_global(script, lang):
    sents = split_sentences(script)
    first = sents[0] if sents else script[:60]
    words = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", script)]
    tags = [w for w, _ in Counter(words).most_common(12)][:8]
    return dict(title=first[:70].rstrip(" .,:;"), description=" ".join(sents[:2])[:300], tags=tags, character="", topic=" ".join(keywords(first, 5)))


def fallback_scene(narration, lang, topic=""):
    kws = keywords(narration, 6)
    subject = ", ".join(kws) if kws else (topic or "concept illustration")
    return dict(image_prompt=f"simple illustration of {subject}", label=short_label(narration, 5))


def analyze_script(narrations, lang, llm: LLM | None, label_mode="ai", translate_to=None, log=print, cancel=None):
    """-> dict(title, description, tags, character, scenes=[dict(narration,image_prompt,on_screen_text)])
    label_mode: 'ai' | 'first words' | 'none'."""
    full = "\n".join(narrations)
    lang_name = LANGS.get(lang, {}).get("name", "English")
    g = fallback_global(full, lang)
    scenes_text = list(narrations)

    if llm is not None:
        try:
            if translate_to and translate_to != lang:
                tname = LANGS[translate_to]["name"]
                log(f"Translating {len(scenes_text)} scene(s) to {tname} ...")
                new = []
                for i, t in enumerate(scenes_text):
                    if cancel is not None and cancel.is_set():
                        raise InterruptedError
                    out = llm.chat(f"You are a professional translator. Translate the user's text to {tname}. Output ONLY the translation.", t,
                                   max_new_tokens=400, temperature=0.2)
                    new.append(out.strip() or t)
                scenes_text, lang_name, lang = new, tname, translate_to
            log("Analysing script (title, topic, recurring character) ...")
            out = llm.chat(
                "You help produce YouTube whiteboard explainer videos. Read the script and answer with exactly these lines:\n"
                f"TITLE: <catchy title, max 70 characters, in {lang_name}>\n"
                f"DESCRIPTION: <2-3 sentence YouTube description in {lang_name}>\n"
                f"TAGS: <8 comma separated tags in {lang_name}>\n"
                "TOPIC: <the topic in 4-8 English words>\n"
                "CHARACTER: <one simple recurring doodle character or mascot that fits the topic, English, max 14 words, or NONE>",
                full[:2500], max_new_tokens=300, temperature=0.5)
            g["title"] = _field(out, "TITLE") or g["title"]
            g["description"] = _field(out, "DESCRIPTION") or g["description"]
            tags = [t.strip() for t in _field(out, "TAGS").split(",") if t.strip()]
            g["tags"] = tags or g["tags"]
            g["topic"] = _field(out, "TOPIC") or g["topic"]
            ch = _field(out, "CHARACTER")
            g["character"] = "" if ch.upper().startswith("NONE") else ch
        except InterruptedError:
            raise
        except Exception as e:
            log(f"! LLM global analysis failed ({e}); using rules")

    scenes = []
    for i, narr in enumerate(scenes_text):
        if cancel is not None and cancel.is_set():
            raise InterruptedError
        sc = fallback_scene(narr, lang, g["topic"])
        if llm is not None:
            try:
                log(f"  scene {i + 1}/{len(scenes_text)}: writing image prompt ...")
                out = llm.chat(
                    "You design visuals for whiteboard explainer videos. For the narration the user gives, reply with exactly two lines:\n"
                    "IMAGE: <ONE concrete, drawable subject in English, max 22 words. Describe objects and simple actions, no abstract words, "
                    "no text or letters inside the picture.>\n"
                    f"LABEL: <a 2-5 word on-screen heading in {lang_name}, or NONE>",
                    f"Video topic: {g['topic']}\nNarration: {narr}", max_new_tokens=90, temperature=0.5)
                img = _field(out, "IMAGE")
                if img and len(img) > 8:
                    sc["image_prompt"] = img.rstrip(".")
                lab = _field(out, "LABEL")
                if lab and not lab.upper().startswith("NONE") and len(lab.split()) <= 8:
                    sc["label"] = lab
            except Exception as e:
                log(f"  ! scene {i + 1}: LLM failed ({e}); using rules")
        if label_mode == "none":
            sc["label"] = ""
        elif label_mode == "first words":
            sc["label"] = short_label(narr, 5)
        scenes.append(dict(narration=narr, image_prompt=sc["image_prompt"], on_screen_text=sc["label"]))
    return dict(title=g["title"], description=g["description"], tags=g["tags"], character=g["character"],
                topic=g["topic"], lang=lang, scenes=scenes)
