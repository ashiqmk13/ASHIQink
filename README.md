# ✏️ InkReel — script → whiteboard-animation YouTube video, 100 % local

Paste a script. InkReel splits it into scenes, writes image prompts, records a voice-over, generates pictures,
**draws them with a moving hand** (outline first, then colour), **handwrites the headings**, adds captions/music and
exports a **YouTube-ready MP4 + thumbnail + subtitles + title/description/chapters/tags**.

* No API keys, no accounts, no paid services — every model is open-source and downloaded **after install** into `./data/`.
* Works on **CPU** (slower) and **GPU** (faster / bigger models). Google Colab notebook included for PCs without a GPU.
* Multi-language voice-over + handwritten text in 30+ languages (incl. Hindi, Malayalam, Tamil, Arabic, Chinese …).

Inspired by Inkplainer (Apache-2.0); the hand/pen images come from it (see `NOTICE`). All code here is new.

---
## 1. Install

Python **3.10 – 3.12** required.

```bash
python install.py                   # auto-detects NVIDIA GPU, makes .venv, installs, downloads CPU-friendly models
./run.sh                            # Windows: run.bat     -> opens http://127.0.0.1:7860
```
Options: `--cpu` / `--gpu`, `--profile minimal|cpu|gpu`, `--no-models` (models then download on first use),
`--with-chatterbox`, `--extra-langs` (Kokoro Japanese+Chinese), `--with-llamacpp` (faster CPU LLM).

**No GPU / weak PC → Google Colab:** open `colab/InkReel_Colab.ipynb` in Colab, run the cells in order, open the `gradio.live` link.

## 2. Use
**Studio tab:** paste script → **⚡ Full auto**. Or *Analyze* → edit the scene table (narration / image prompt / on-screen text) → *Generate*.
Outputs land in `projects/<date>-<title>/`: `<title>.mp4`, `thumbnail.jpg`, `subtitles.srt`, `youtube.txt`.

| You can set | Where |
|---|---|
| Style preset, extra style words, **negative prompt**, seed, steps | Look & AI pictures |
| **Reference image(s) / reference video** (video → key-frames) | Look & AI pictures (IP-Adapter on SD1.5/SDXL, palette on others) |
| Your own images instead of AI | Look & AI pictures / *Image → drawing video* tab |
| Draw style (outline→colour, scanner, chunk-jump), hand/pen, custom hand PNG | Animation & text |
| **Fonts**: upload `.ttf/.otf`, or download Google/Noto fonts | Fonts tab |
| Captions, zoom, eraser wipe, music, 480p/720p/1080p, 16:9 / 9:16 / 1:1 | Animation / Output |

CLI: `python -m inkreel.cli examples/sample_script.txt --res 1080p --style "Cartoon explainer"`

## 3. What runs what

**Voice (TTS)** — *Auto* picks the first engine that supports your language.

| Engine | Languages | CPU? | Licence of weights |
|---|---|---|---|
| Kokoro 82M | EN-US/UK, ES, FR, HI, IT, PT-BR, JA, ZH | ✅ fast | Apache-2.0 |
| Chatterbox (+ voice cloning) | 23: ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh | ⚠️ slow, GPU better | MIT |
| MMS-TTS | 1000+ incl. **ml ta te kn bn mr gu pa ur** | ✅ | **CC-BY-NC (non-commercial!)** |
| espeak-ng | 100+ (robotic) | ✅ no download | GPL (system tool) |

> Monetising on YouTube? Avoid MMS-TTS; use Kokoro / Chatterbox / your own recorded voice (*Image → drawing video* tab accepts an audio file).

**Script brain (LLM):** Qwen2.5 0.5B / 1.5B / 7B (Apache-2.0) via transformers, or llama.cpp if installed. Your narration is **never rewritten** (unless you pick "translate"); the LLM only adds image prompts, headings, title, description, tags. "None" = rule-based fallback.

**Images:** SD 1.5 + LCM (default, CPU-friendly, 5 steps) · SDXL + LCM (GPU) · FLUX.1-schnell (≥16 GB GPU) · *Instant doodles* (no AI; for testing timing) · or any SD1.5 `.safetensors` checkpoint.

**Drawing engine:** line-art extraction → skeleton "pen strokes" (follows centre-lines, continues through junctions) → hatch fill for thick areas → colour brush → hand sprite with enter/exit, jitter, zoom and eraser wipe. Text is drawn glyph by glyph in reading order.

## 4. Speed expectations (CPU, 8 cores, 720p)
Voice (Kokoro) ≈ real-time · image (SD1.5-LCM 640×360) ≈ 15-60 s · LLM 1.5B ≈ 10 s/scene · frame rendering + encode ≈ faster than real-time. A 10-scene video ≈ 10-20 min on CPU, 2-4 min on a mid GPU.

## 5. Honest status / limitations
* The drawing engine, text/fonts, captions, ffmpeg export, chapters/thumbnail, UI wiring and the placeholder doodle mode were **run and checked end-to-end** while building this. The AI back-ends (Kokoro, Chatterbox, MMS, diffusers, Qwen, IP-Adapter) are written against their documented APIs but could **not be executed in the build sandbox (no internet/GPU)** — expect to fix a version quirk or two on first run; errors are shown in the UI log, and *Auto* falls back to another engine when one fails.
* Font download URLs (Google Fonts / Noto) were not testable offline. If one fails, drop any `.ttf` into `fonts/` or use the Fonts tab.
* Indic/Arabic text needs Pillow's RAQM shaper (`apt install libraqm0` on Linux).
* Chatterbox pins old torch versions, so it lives in its own `.venv_chatterbox` (created by `--with-chatterbox`) and talks to the app through a worker process.
* Small LLMs are weaker in Malayalam/Tamil etc.: headings/translation may need manual editing in the scene table. Image prompts are always English.
* AI pictures are not guaranteed to be clean line art; the *Whiteboard* presets + negative prompt help, and the *photo* fallback turns any picture into a pencil-sketch outline.
