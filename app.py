"""InkReel - local web UI.   python app.py [--share] [--port 7860]"""
import argparse
import queue
import threading
import time
import traceback
from pathlib import Path

import gradio as gr

import inkreel  # noqa: F401  (sets HF cache paths)
from inkreel import fonts, hardware, hands, prefetch, tts
from inkreel.config import ASPECTS, RESOLUTIONS, PROJECTS_DIR
from inkreel.imagegen import IMAGE_CHOICES, IMAGE_MODELS
from inkreel.langs import LANG_CHOICES, LANGS
from inkreel.llm import LLM_CHOICES
from inkreel.pipeline import Cancelled, Settings, analyze, images_to_video, render
from inkreel.styles import STYLE_NAMES

CANCEL = threading.Event()

CSS = """
:root { 
  --ink: #6c5ce7; 
  --ink2: #00b894; 
  --bg-dark: #0f172a;
}
.gradio-container { max-width: 1380px !important; margin: auto; }
#hero { 
  padding: 22px 28px; 
  border-radius: 18px; 
  margin-bottom: 12px;
  background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #0f766e 100%); 
  color: #fff; 
  box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.3);
}
#hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: .5px; }
#hero p { margin: 6px 0 0; opacity: .9; font-size: 1.05rem; }
.bigbtn button, button.bigbtn { font-size: 1.05rem !important; padding: 14px 10px !important; font-weight: 600; }
#log textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
.note { font-size: .85rem; opacity: .8; }

/* Dashboard & Progress Styling */
.dashboard-card {
  background: rgba(15, 23, 42, 0.6);
  border: 1px solid #334155;
  border-radius: 14px;
  padding: 18px;
  margin-bottom: 16px;
  color: #f8fafc;
}
.dash-stats-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  flex-wrap: wrap;
  gap: 10px;
}
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 6px 14px;
  border-radius: 20px;
  background: rgba(59, 130, 246, 0.15);
  color: #60a5fa;
  font-weight: 600;
  font-size: 0.95rem;
  border: 1px solid rgba(59, 130, 246, 0.3);
}
.pulse-indicator {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background-color: #3b82f6;
  box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7);
  animation: pulse-ring 1.5s infinite cubic-bezier(0.66, 0, 0, 1);
}
@keyframes pulse-ring {
  to { box-shadow: 0 0 0 10px rgba(59, 130, 246, 0); }
}
.time-box {
  display: flex;
  gap: 16px;
  font-family: ui-monospace, Menlo, Consolas, monospace;
  font-size: 0.92rem;
  color: #cbd5e1;
  background: #1e293b;
  padding: 6px 14px;
  border-radius: 8px;
  border: 1px solid #334155;
}
.progress-outer {
  width: 100%;
  height: 24px;
  background: #1e293b;
  border-radius: 12px;
  overflow: hidden;
  position: relative;
  border: 1px solid #334155;
  margin-bottom: 16px;
}
.progress-inner {
  height: 100%;
  background: linear-gradient(90deg, #4f46e5 0%, #06b6d4 50%, #10b981 100%);
  transition: width 0.4s ease-in-out;
}
.progress-percentage {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  text-align: center;
  line-height: 24px;
  font-weight: 700;
  font-size: 0.85rem;
  color: #ffffff;
  text-shadow: 0 1px 3px rgba(0,0,0,0.8);
}
.tasks-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 10px;
  margin-top: 8px;
}
.task-item {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 10px;
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.task-item.active {
  border-color: #3b82f6;
  background: rgba(59, 130, 246, 0.12);
}
.task-item.completed {
  border-color: #10b981;
  background: rgba(16, 185, 129, 0.12);
}
.task-title {
  font-weight: 600;
  font-size: 0.86rem;
  color: #f1f5f9;
}
.task-status-text {
  font-size: 0.78rem;
  color: #94a3b8;
}

/* Download Center Styling */
.download-center-card {
  background: rgba(15, 23, 42, 0.75);
  border: 1px solid #3b82f6;
  border-radius: 14px;
  padding: 18px;
  margin-top: 16px;
  color: #f8fafc;
}
.dl-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
  margin-top: 12px;
}
.dl-card {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 10px;
  padding: 12px 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.dl-card:hover {
  border-color: #3b82f6;
}
.dl-info {
  display: flex;
  align-items: center;
  gap: 10px;
}
.dl-icon { font-size: 1.4rem; }
.dl-title { font-weight: 700; color: #f8fafc; font-size: 0.92rem; }
.dl-desc { font-size: 0.76rem; color: #94a3b8; }
.dl-button {
  background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
  color: white !important;
  padding: 8px 14px;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.85rem;
  text-decoration: none !important;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  white-space: nowrap;
}
.dl-button:hover {
  background: linear-gradient(135deg, #4338ca 0%, #2563eb 100%);
}
"""

LANG_NO_AUTO = [c for c in LANG_CHOICES if c[1] != "auto"]
SETTING_KEYS = []
S = {}


def comp(key, c):
    S[key] = c
    SETTING_KEYS.append(key)
    return c


def mk_settings(vals) -> Settings:
    d = dict(zip(SETTING_KEYS, vals))
    tip = (float(d.pop("hand_tip_x") or 0.5), float(d.pop("hand_tip_y") or 0.0))
    d["custom_hand_tip"] = tip
    d["hand_scale"] = float(d.get("hand_scale") or 0.0)
    for k in ("ref_images", "user_images"):
        d[k] = d.get(k) or []
    d["seed"] = int(d.get("seed") if d.get("seed") is not None else 1234)
    d["image_steps"] = int(d.get("image_steps") or 0)
    d["fps"] = int(d.get("fps") or 30)
    return Settings(**d)


def render_dashboard_html(pct_float, step_msg, elapsed_sec):
    pct = max(0, min(100, int(pct_float * 100)))
    
    if pct_float > 0.01 and pct_float < 1.0:
        total_est = elapsed_sec / pct_float
        eta_sec = max(0, int(total_est - elapsed_sec))
        eta_str = f"{eta_sec // 60:02d}:{eta_sec % 60:02d}"
    elif pct_float >= 1.0:
        eta_str = "00:00 (Done)"
    else:
        eta_str = "Calculating..."
        
    elapsed_str = f"{int(elapsed_sec) // 60:02d}:{int(elapsed_sec) % 60:02d}"
    
    stages = [
        ("Script Analysis", 0.0, 0.02, "📝"),
        ("Voice Synthesis", 0.02, 0.20, "🗣️"),
        ("AI Image Gen", 0.20, 0.55, "🎨"),
        ("Scene Animation", 0.55, 0.94, "✍️"),
        ("Final Packaging", 0.94, 1.00, "🎬"),
    ]
    
    stage_htmls = []
    for name, start_f, end_f, icon in stages:
        if pct_float >= end_f or (end_f == 1.0 and pct_float >= 1.0):
            st_cls = "completed"
            st_lbl = "✓ Done"
        elif pct_float >= start_f:
            st_cls = "active"
            st_lbl = f"⏳ In Progress ({pct}%)"
        else:
            st_cls = "queued"
            st_lbl = "🕒 Upcoming"
            
        stage_htmls.append(f"""
        <div class="task-item {st_cls}">
          <div class="task-title">{icon} {name}</div>
          <div class="task-status-text">{st_lbl}</div>
        </div>
        """)
        
    tasks_html = "".join(stage_htmls)
    badge_anim = '<div class="pulse-indicator"></div>' if pct < 100 else '✓'
    
    return f"""
    <div class="dashboard-card">
      <div class="dash-stats-row">
        <div class="status-badge">
          {badge_anim}
          <span>{step_msg or 'Ready'}</span>
        </div>
        <div class="time-box">
          <span>⏱️ Elapsed: <strong>{elapsed_str}</strong></span>
          <span>⏳ Time Left: <strong>{eta_str}</strong></span>
        </div>
      </div>
      <div class="progress-outer">
        <div class="progress-inner" style="width: {pct}%;"></div>
        <div class="progress-percentage">{pct}% Completed</div>
      </div>
      <div style="font-weight: 600; font-size: 0.88rem; margin-bottom: 6px; color: #94a3b8;">
        📋 Tasks Coming & Pipeline Stages:
      </div>
      <div class="tasks-grid">
        {tasks_html}
      </div>
    </div>
    """


def build_download_center_html(out_dict):
    if not out_dict:
        return ""
    
    items = []
    labels = {
        "video": ("🎬", "Final Video MP4", "Primary whiteboard animation video"),
        "thumbnail": ("🖼️", "Thumbnail JPG", "High-res YouTube thumbnail"),
        "subtitles": ("📝", "Subtitles SRT", "Timed caption file"),
        "youtube": ("📄", "YouTube Metadata TXT", "Title, description, chapters & tags"),
        "zip": ("📦", "All Assets ZIP Bundle", "Download everything in 1 ZIP file"),
    }
    
    for key, (icon, title, desc) in labels.items():
        if key in out_dict and out_dict[key] and Path(out_dict[key]).exists():
            fp = out_dict[key]
            web_path = f"/file={fp}"
            fname = Path(fp).name
            items.append(f"""
            <div class="dl-card">
              <div class="dl-info">
                <span class="dl-icon">{icon}</span>
                <div>
                  <div class="dl-title">{title}</div>
                  <div class="dl-desc">{desc} ({fname})</div>
                </div>
              </div>
              <a href="{web_path}" download="{fname}" target="_blank" class="dl-button">
                ⬇️ Download
              </a>
            </div>
            """)
            
    if not items:
        return ""
        
    return f"""
    <div class="download-center-card">
      <h3 style="margin-top:0; margin-bottom: 10px; color:#f8fafc; font-size: 1.1rem;">📥 Download Center (Click Any File to Start Download)</h3>
      <div class="dl-grid">
        {"".join(items)}
      </div>
    </div>
    """


def run_bg(fn):
    """Run fn(log, progress) in a thread, yield (log_text, dashboard_html, status_text, result_dict) as it goes."""
    q = queue.Queue()
    res = {}
    t_start = time.time()

    def target():
        try:
            res["out"] = fn(lambda m: q.put(("log", str(m))), lambda f, d="": q.put(("prog", (f, d))))
        except Cancelled:
            res["cancelled"] = True
            q.put(("log", "Cancelled by user."))
        except Exception as e:
            res["err"] = str(e)
            q.put(("log", "ERROR: " + str(e) + "\n" + traceback.format_exc()))
        finally:
            q.put(("done", None))

    threading.Thread(target=target, daemon=True).start()
    lines = []
    status_text = "Starting ..."
    prog_f = 0.0
    
    while True:
        elapsed = time.time() - t_start
        try:
            kind, val = q.get(timeout=0.4)
        except queue.Empty:
            dash_html = render_dashboard_html(prog_f, status_text, elapsed)
            yield "\n".join(lines[-400:]), dash_html, status_text, None
            continue
        if kind == "log":
            lines.append(val)
        elif kind == "prog":
            prog_f = float(val[0])
            status_text = f"{int(prog_f * 100)}% - {val[1]}"
        else:
            break
        dash_html = render_dashboard_html(prog_f, status_text, elapsed)
        yield "\n".join(lines[-400:]), dash_html, status_text, None
        
    elapsed = time.time() - t_start
    final_f = 1.0 if "out" in res else prog_f
    final_status = "Done" if "out" in res else "Stopped / failed - see log"
    dash_html = render_dashboard_html(final_f, final_status, elapsed)
    yield "\n".join(lines[-400:]), dash_html, final_status, res


def rows_from_scenes(scenes):
    return [[s["narration"], s["image_prompt"], s["on_screen_text"]] for s in scenes]


def scenes_from_rows(rows):
    out = []
    for r in rows or []:
        if r and str(r[0]).strip() and str(r[0]).strip().lower() != "nan":
            out.append(dict(narration=str(r[0]).strip(), image_prompt=str(r[1] or "").strip() if len(r) > 1 else "",
                            on_screen_text=str(r[2] or "").strip() if len(r) > 2 else ""))
    return out


# ----------------------------------------------------------------------------- handlers
def do_analyze(*vals):
    CANCEL.clear()
    st = mk_settings(vals)
    for log, dash, status, res in run_bg(lambda lg, pg: analyze(st, lg, CANCEL)):
        if res is None:
            yield log, dash, status, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        elif "out" in res:
            m = res["out"]
            yield (log, dash, status, rows_from_scenes(m["scenes"]), m["title"], m["description"], ", ".join(m["tags"]), m)
        else:
            yield log, dash, status, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()


def _render_gen(st, meta, rows, title, desc, tags):
    scenes = scenes_from_rows(rows)
    meta = dict(meta or {})
    meta.update(title=title or meta.get("title", "video"), description=desc, tags=[t.strip() for t in (tags or "").split(",") if t.strip()])
    meta.setdefault("lang", None)

    def job(lg, pg):
        return render(st, meta, scenes, None, lg, pg, CANCEL)
    return run_bg(job)


def _render_outputs(log, dash, status, res):
    if res is None or "out" not in res:
        return log, dash, status, gr.update(), gr.update(), gr.update(), ""
    o = res["out"]
    files = [o["video"], o["thumbnail"], o["subtitles"], o["youtube"]]
    if "zip" in o and o["zip"]:
        files.append(o["zip"])
    dl_html = build_download_center_html(o)
    return log, dash, status, o["video"], o["thumbnail"], files, dl_html


def do_render(*args):
    CANCEL.clear()
    *vals, meta, rows, title, desc, tags = args
    st = mk_settings(vals)
    if not rows or not scenes_from_rows(rows):
        yield "No scenes yet - click 'Analyze script' first (or use Full auto).", render_dashboard_html(0, "Nothing to do", 0), "Nothing to do", gr.update(), gr.update(), gr.update(), ""
        return
    for log, dash, status, res in _render_gen(st, meta, rows, title, desc, tags):
        yield _render_outputs(log, dash, status, res)


def do_full_auto(*vals):
    CANCEL.clear()
    st = mk_settings(vals)
    log_acc = ""
    meta = None
    for log, dash, status, res in run_bg(lambda lg, pg: analyze(st, lg, CANCEL)):
        if res is not None and "out" in res:
            meta = res["out"]
        yield log, dash, status, gr.update(), gr.update(), gr.update(), "", gr.update(), gr.update(), gr.update(), gr.update()
        log_acc = log
    if meta is None:
        return
    rows = rows_from_scenes(meta["scenes"])
    tags = ", ".join(meta["tags"])
    yield log_acc + "\n--- rendering ---", render_dashboard_html(0.02, "Rendering ...", 0), "Rendering ...", gr.update(), gr.update(), gr.update(), "", rows, meta["title"], meta["description"], tags
    for log, dash, status, res in _render_gen(st, meta, rows, meta["title"], meta["description"], tags):
        l, d, s, v, t, f, dl = _render_outputs(log_acc + "\n--- rendering ---\n" + log, dash, status, res)
        yield l, d, s, v, t, f, dl, rows, meta["title"], meta["description"], tags


def do_stop():
    CANCEL.set()
    return render_dashboard_html(0, "Stopping after current step...", 0), "Stopping after the current step ..."


def do_quick(files, sec, style, mode, hand, aspect, res, color, audio):
    CANCEL.clear()
    if not files:
        yield "Upload at least one image.", render_dashboard_html(0, "No images uploaded", 0), "", None, [], ""
        return
    st = Settings(style=style, anim_mode=mode, hand=hand, aspect=aspect, resolution=res, use_color=color, fps=30, transition="none")
    paths = [f if isinstance(f, str) else f.name for f in files]
    for log, dash, status, r in run_bg(lambda lg, pg: images_to_video(st, paths, float(sec), audio, lg, pg, CANCEL)):
        out_vid = r["out"] if r and "out" in r else None
        files_out = [out_vid] if out_vid else []
        dl_dict = {"video": out_vid} if out_vid else {}
        dl_html = build_download_center_html(dl_dict) if out_vid else ""
        yield log, dash, status, out_vid, files_out, dl_html


def voices_update(engine, lang):
    v = tts.voices_for(engine, lang) if lang != "auto" else []
    return gr.update(choices=["auto"] + v, value="auto")


def fonts_refresh():
    return gr.update(choices=["Auto"] + fonts.list_fonts())


def font_add(files):
    for f in files or []:
        fonts.add_font(f if isinstance(f, str) else f.name)
    return fonts_refresh(), fonts_refresh(), f"Fonts available: {', '.join(fonts.list_fonts()) or 'none yet'}"


def font_download(keys):
    lines = []
    fonts.ensure_fonts(keys or fonts.BASIC_KEYS, lambda m: lines.append(str(m)))
    return fonts_refresh(), fonts_refresh(), "\n".join(lines) + f"\nAvailable: {', '.join(fonts.list_fonts())}"


def font_preview(name, lang, text):
    script = LANGS.get(lang, LANGS["en"])["script"]
    return fonts.preview(name, text or "The quick brown fox", script)


def model_download(llm, img, engines, tts_lang, font_keys):
    CANCEL.clear()

    def job(lg, pg):
        prefetch.run(llm=llm, image=img, tts=engines, tts_lang=tts_lang, font_keys=font_keys, log=lg)
        return True
    for log, dash, status, _ in run_bg(job):
        yield log, dash, status


def disk_usage():
    return prefetch.disk_report()


def list_projects():
    if not PROJECTS_DIR.exists():
        return []
    projs = [d.name for d in PROJECTS_DIR.iterdir() if d.is_dir()]
    projs.sort(reverse=True)
    return projs or ["No projects created yet"]


def load_project_details(proj_name):
    if not proj_name or proj_name == "No projects created yet":
        return None, None, [], ""
    p = PROJECTS_DIR / proj_name
    if not p.exists():
        return None, None, [], ""
    
    mp4s = list(p.glob("*.mp4"))
    video_path = str(mp4s[0]) if mp4s else None
    
    thumbs = list(p.glob("*.jpg"))
    thumb_path = str(thumbs[0]) if thumbs else None
    
    files = [str(f) for f in p.glob("*") if f.is_file()]
    
    out_dict = {}
    if video_path: out_dict["video"] = video_path
    if thumb_path: out_dict["thumbnail"] = thumb_path
    srts = list(p.glob("*.srt"))
    if srts: out_dict["subtitles"] = str(srts[0])
    txts = list(p.glob("*.txt"))
    if txts: out_dict["youtube"] = str(txts[0])
    zips = list(p.glob("*.zip"))
    if zips: out_dict["zip"] = str(zips[0])
    
    dl_html = build_download_center_html(out_dict)
    return video_path, thumb_path, files, dl_html


# ----------------------------------------------------------------------------- UI
def build():
    with gr.Blocks(title="InkReel - AI whiteboard video studio", css=CSS) as demo:
        gr.HTML("<div id='hero'><h1>✏️ InkReel Studio</h1><p>Script → voice-over → AI pictures → hand-drawn whiteboard animation → YouTube-ready MP4. "
                "100% local · no API · no subscription.</p></div>")
        gr.Markdown(f"<span class='note'>🖥️ {hardware.describe()}</span>")
        meta_state = gr.State({})

        with gr.Tabs():
            # ------------------------------------------------------------ STUDIO
            with gr.Tab("🎬 Studio"):
                with gr.Row():
                    with gr.Column(scale=5):
                        comp("script", gr.Textbox(label="Your script", lines=11, placeholder="Paste your full script here. Blank lines = scene breaks (optional). Any supported language."))
                        with gr.Row():
                            comp("title", gr.Textbox(label="Video title (optional - AI suggests one)", scale=3))
                            comp("lang", gr.Dropdown(LANG_CHOICES, value="auto", label="Script language", scale=2))
                        with gr.Row():
                            full_btn = gr.Button("⚡ Full auto: script → final video", variant="primary", elem_classes="bigbtn")
                        with gr.Row():
                            an_btn = gr.Button("1 · Analyze script (review scenes first)", elem_classes="bigbtn")
                            rd_btn = gr.Button("2 · Generate video from scenes", elem_classes="bigbtn")
                            stop_btn = gr.Button("⏹ Stop", variant="stop")
                    with gr.Column(scale=6):
                        with gr.Accordion("🗣️ Voice", open=True):
                            with gr.Row():
                                comp("tts_engine", gr.Dropdown(tts.ENGINE_CHOICES, value="auto", label="Voice engine"))
                                comp("voice", gr.Dropdown(["auto"], value="auto", label="Voice (Kokoro)", allow_custom_value=True))
                            with gr.Row():
                                comp("speech_speed", gr.Slider(0.7, 1.4, value=1.0, step=0.05, label="Speaking speed"))
                                comp("ref_voice", gr.Audio(type="filepath", label="Voice to clone (Chatterbox only, 5-15 s wav)"))
                        with gr.Accordion("🎨 Look & AI pictures", open=True):
                            comp("style", gr.Dropdown(STYLE_NAMES, value=STYLE_NAMES[1], label="Style preset"))
                            comp("style_extra", gr.Textbox(label="Extra style words (optional)", placeholder="e.g. blue and orange palette, thick marker, cute"))
                            comp("negative", gr.Textbox(label="Negative prompt (what to avoid)", placeholder="e.g. people, faces, shadows, clutter"))
                            comp("image_model", gr.Dropdown(IMAGE_CHOICES, value="sd15-lcm", label="Image model"))
                            with gr.Row():
                                comp("image_steps", gr.Slider(0, 20, value=0, step=1, label="Steps (0 = model default)"))
                                comp("seed", gr.Number(value=1234, precision=0, label="Seed (-1 = random)"))
                            comp("use_color", gr.Radio([("From style", "style"), ("Always colour in", "yes"), ("Line art only", "no")], value="style", label="Colour fill"))
                            with gr.Row():
                                comp("ref_images", gr.File(label="Reference style image(s)", file_count="multiple", type="filepath", file_types=["image"]))
                                comp("ref_video", gr.Video(label="Reference video (key-frames → style)"))
                            comp("ref_strength", gr.Slider(0.1, 1.0, value=0.6, step=0.05, label="Reference strength (IP-Adapter, SD1.5 / SDXL)"))
                            comp("user_images", gr.File(label="Use my own images instead of AI (scene 1, 2, 3 ... by file name)", file_count="multiple", type="filepath", file_types=["image"]))
                            comp("custom_checkpoint", gr.Textbox(label="Custom SD1.5 checkpoint (.safetensors path, optional)"))
                        with gr.Accordion("✋ Animation & text", open=False):
                            comp("anim_mode", gr.Radio([("Draw: outline → colour", "draw"), ("Scanner", "scanner"), ("Chunk jump", "chunks")], value="draw", label="Drawing style"))
                            with gr.Row():
                                comp("hand", gr.Dropdown(hands.HAND_CHOICES, value="hand1", label="Hand / pen"))
                                comp("hand_scale", gr.Slider(0, 1.0, value=0, step=0.05, label="Hand size (0 = auto)"))
                            with gr.Row():
                                comp("custom_hand", gr.File(label="Custom hand PNG (choose 'Custom upload')", type="filepath", file_types=[".png"]))
                                comp("hand_tip_x", gr.Slider(0, 1, value=0.5, step=0.01, label="Pen tip X (0-1 of image width)"))
                                comp("hand_tip_y", gr.Slider(0, 1, value=0.0, step=0.01, label="Pen tip Y (0-1 of image height)"))
                            comp("draw_ratio", gr.Slider(0.3, 1.0, value=0.8, step=0.05, label="Drawing time (share of the narration length)"))
                            comp("label_mode", gr.Radio([("AI key phrase", "ai"), ("First words of narration", "first words"), ("No on-screen text", "none")], value="ai", label="On-screen handwritten text"))
                            with gr.Row():
                                comp("font", gr.Dropdown(["Auto"] + fonts.list_fonts(), value="Auto", label="Font (upload more in Fonts tab)"))
                                comp("text_color", gr.Textbox(label="Text colour hex (blank = style default)", placeholder="#1a1a1a"))
                            with gr.Row():
                                comp("text_first", gr.Checkbox(value=True, label="Write heading before drawing picture"))
                                comp("zoom", gr.Checkbox(value=True, label="Slow zoom while holding"))
                                comp("captions", gr.Checkbox(value=False, label="Burn-in captions"))
                            comp("transition", gr.Radio([("Eraser wipe", "wipe"), ("Hard cut", "none")], value="wipe", label="Scene transition"))
                        with gr.Accordion("📦 Output & performance", open=False):
                            with gr.Row():
                                comp("aspect", gr.Dropdown(list(ASPECTS), value="16:9 (YouTube)", label="Aspect"))
                                comp("resolution", gr.Dropdown(list(RESOLUTIONS), value="720p", label="Resolution"))
                                comp("fps", gr.Radio([24, 30], value=30, label="FPS"))
                            with gr.Row():
                                comp("device", gr.Radio([("Auto", "auto"), ("CPU only", "cpu"), ("GPU (CUDA)", "cuda")], value="auto", label="Compute"))
                                comp("llm_model", gr.Dropdown(LLM_CHOICES, value=LLM_CHOICES[1], label="Script-analysis LLM"))
                            comp("translate_to", gr.Dropdown([("No translation", "none")] + LANG_NO_AUTO, value="none", label="Translate script to (needs an LLM; quality varies by language)"))
                            comp("scene_sec", gr.Slider(4, 20, value=9, step=1, label="Target scene length (seconds)"))
                            with gr.Row():
                                comp("music", gr.Audio(type="filepath", label="Background music (optional)"))
                                comp("music_vol", gr.Slider(0.02, 0.5, value=0.12, step=0.01, label="Music volume"))

                gr.Markdown("### Scenes  <span class='note'>(edit narration / image prompt / on-screen text; add or delete rows; empty prompt = text-only scene)</span>")
                scenes_df = gr.Dataframe(headers=["Narration", "Image prompt (English)", "On-screen text"], datatype=["str", "str", "str"],
                                         column_count=(3, "fixed"), row_count=(1, "dynamic"), interactive=True, wrap=True, type="array")
                with gr.Row():
                    meta_title = gr.Textbox(label="YouTube title", scale=3)
                    meta_tags = gr.Textbox(label="Tags (comma separated)", scale=3)
                meta_desc = gr.Textbox(label="YouTube description", lines=2)

                # Dashboard Progress Header with Tasks Coming & Time Left
                dash_box = gr.HTML(value=render_dashboard_html(0, "Ready to start", 0))
                status = gr.Textbox(label="Status Summary", interactive=False)
                log_box = gr.Textbox(label="Live Execution Log", lines=9, max_lines=16, interactive=False, elem_id="log")
                
                with gr.Row():
                    out_video = gr.Video(label="Final video preview", scale=3)
                    out_thumb = gr.Image(label="YouTube thumbnail", scale=2)
                    
                out_files = gr.File(label="📥 Download Center: Click any file to download (Video · Thumbnail · Subtitles · Metadata · ZIP Bundle)", file_count="multiple")
                out_download_center = gr.HTML()

                vals = [S[k] for k in SETTING_KEYS]
                an_btn.click(do_analyze, vals, [log_box, dash_box, status, scenes_df, meta_title, meta_desc, meta_tags, meta_state])
                rd_btn.click(do_render, vals + [meta_state, scenes_df, meta_title, meta_desc, meta_tags],
                             [log_box, dash_box, status, out_video, out_thumb, out_files, out_download_center])
                full_btn.click(do_full_auto, vals, [log_box, dash_box, status, out_video, out_thumb, out_files, out_download_center, scenes_df, meta_title, meta_desc, meta_tags])
                stop_btn.click(do_stop, None, [dash_box, status])
                S["lang"].change(voices_update, [S["tts_engine"], S["lang"]], S["voice"])
                S["tts_engine"].change(voices_update, [S["tts_engine"], S["lang"]], S["voice"])

            # ------------------------------------------------------------ QUICK TOOL
            with gr.Tab("🖼️ Image → drawing video"):
                gr.Markdown("Turn your own pictures into hand-drawn whiteboard animations (no AI models needed).")
                with gr.Row():
                    with gr.Column():
                        q_files = gr.File(label="Images", file_count="multiple", type="filepath", file_types=["image"])
                        q_audio = gr.Audio(type="filepath", label="Audio / voice-over (optional)")
                        q_sec = gr.Slider(2, 30, value=6, step=1, label="Seconds per image")
                        q_style = gr.Dropdown(STYLE_NAMES, value=STYLE_NAMES[1], label="Board style")
                        q_color = gr.Radio([("From style", "style"), ("Colour", "yes"), ("Line art", "no")], value="style", label="Colour fill")
                        q_mode = gr.Radio([("Draw", "draw"), ("Scanner", "scanner"), ("Chunk jump", "chunks")], value="draw", label="Drawing style")
                        q_hand = gr.Dropdown(hands.HAND_CHOICES, value="hand1", label="Hand")
                        with gr.Row():
                            q_aspect = gr.Dropdown(list(ASPECTS), value="16:9 (YouTube)", label="Aspect")
                            q_res = gr.Dropdown(list(RESOLUTIONS), value="720p", label="Resolution")
                        q_go = gr.Button("Create drawing video", variant="primary", elem_classes="bigbtn")
                    with gr.Column():
                        q_dash = gr.HTML(value=render_dashboard_html(0, "Ready", 0))
                        q_status = gr.Textbox(label="Status", interactive=False)
                        q_log = gr.Textbox(label="Log", lines=6, interactive=False)
                        q_out = gr.Video(label="Result Video")
                        q_files_out = gr.File(label="📥 Download Result File", file_count="multiple")
                        q_download_center = gr.HTML()
                q_go.click(do_quick, [q_files, q_sec, q_style, q_mode, q_hand, q_aspect, q_res, q_color, q_audio],
                           [q_log, q_dash, q_status, q_out, q_files_out, q_download_center])

            # ------------------------------------------------------------ PROJECTS & DOWNLOADS
            with gr.Tab("📁 Projects & Downloads"):
                gr.Markdown("Browse all completed projects and click any file or ZIP archive to download immediately.")
                with gr.Row():
                    proj_dropdown = gr.Dropdown(choices=list_projects(), label="Select Project Folder", interactive=True)
                    proj_refresh = gr.Button("🔄 Refresh List")
                with gr.Row():
                    p_video = gr.Video(label="Video Preview", scale=3)
                    p_thumb = gr.Image(label="Thumbnail", scale=2)
                p_files = gr.File(label="Project Files (Click to Download)", file_count="multiple")
                p_dl_html = gr.HTML()

                proj_dropdown.change(load_project_details, proj_dropdown, [p_video, p_thumb, p_files, p_dl_html])
                proj_refresh.click(lambda: gr.update(choices=list_projects()), None, proj_dropdown)

            # ------------------------------------------------------------ FONTS
            with gr.Tab("🔤 Fonts"):
                gr.Markdown("Text is drawn with a **handwriting-style pen** using these fonts. 'Auto' picks a font that supports your script "
                            "(Latin, Devanagari, Malayalam, Tamil, Arabic, CJK ...). Upload any `.ttf` / `.otf` to use your own.")
                with gr.Row():
                    f_up = gr.File(label="Upload fonts", file_count="multiple", type="filepath", file_types=[".ttf", ".otf", ".ttc"])
                    f_keys = gr.CheckboxGroup([(f"{k}  -  {v[2]}", k) for k, v in fonts.FONT_CATALOG.items()], value=fonts.BASIC_KEYS, label="Download open-source fonts (Google Fonts / Noto, OFL licence)")
                with gr.Row():
                    f_add = gr.Button("Add uploaded fonts"); f_dl = gr.Button("Download selected fonts")
                f_log = gr.Textbox(label="Fonts", lines=4, interactive=False, value=", ".join(fonts.list_fonts()))
                with gr.Row():
                    f_sel = gr.Dropdown(["Auto"] + fonts.list_fonts(), value="Auto", label="Preview font")
                    f_lang = gr.Dropdown(LANG_NO_AUTO, value="en", label="Language")
                    f_txt = gr.Textbox(value="The quick brown fox jumps", label="Preview text")
                f_img = gr.Image(label="Preview")
                f_add.click(font_add, f_up, [S["font"], f_sel, f_log])
                f_dl.click(font_download, f_keys, [S["font"], f_sel, f_log])
                for c in (f_sel, f_lang, f_txt):
                    c.change(font_preview, [f_sel, f_lang, f_txt], f_img)

            # ------------------------------------------------------------ MODELS
            with gr.Tab("📥 Models & setup"):
                gr.Markdown("Nothing is bundled: AI models download on first use into the `data/` folder. "
                            "Pre-download here so the first video doesn't wait.")
                rec = hardware.recommend()
                gr.Markdown(f"**This PC:** {hardware.describe()}  \n**Suggested:** image model `{rec['image_model']}`, LLM *{rec['llm']}* - {rec['note']}")
                with gr.Row():
                    m_llm = gr.Dropdown(["(skip)"] + [k for k in LLM_CHOICES if "None" not in k], value="(skip)", label="Script LLM")
                    m_img = gr.Dropdown([("(skip)", "skip")] + [(v["label"], k) for k, v in IMAGE_MODELS.items() if k != "doodle"], value="skip", label="Image model")
                with gr.Row():
                    m_tts = gr.CheckboxGroup([("Kokoro", "kokoro"), ("MMS-TTS (non-commercial licence)", "mms"), ("Chatterbox (installed separately)", "chatterbox")], label="Voice engines")
                    m_lang = gr.Dropdown(LANG_NO_AUTO, value="en", label="…for language")
                m_fonts = gr.CheckboxGroup([(k, k) for k in fonts.FONT_CATALOG], value=[], label="Fonts")
                m_go = gr.Button("Download selected", variant="primary")
                m_dash = gr.HTML(value=render_dashboard_html(0, "Ready", 0))
                m_status = gr.Textbox(label="Status", interactive=False)
                m_log = gr.Textbox(label="Log", lines=8, interactive=False)
                m_disk = gr.Textbox(label="Disk usage of data/ folder", value=disk_usage, interactive=False)
                m_go.click(model_download, [m_llm, m_img, m_tts, m_lang, m_fonts], [m_log, m_dash, m_status]).then(disk_usage, None, m_disk)

            # ------------------------------------------------------------ HELP
            with gr.Tab("❓ Help"):
                gr.Markdown(HELP)
    return demo


HELP = """
### Quick start
1. Paste a script → **⚡ Full auto** (or *Analyze* first to edit scenes).
2. Result + thumbnail + subtitles + `youtube.txt` (title / description / chapters / tags) appear below and in `projects/`.

### Tips
* **CPU-only?** Keep *SD1.5 + LCM*, 720p, Qwen 1.5B. Expect ~1-2 min per scene. Use *Instant doodles* to test the timing first.
* **Cached work:** voice and images are cached per project; change drawing/hand/caption settings and re-render quickly by using the same script.
* **Malayalam / Tamil / Telugu …:** Auto engine → MMS-TTS (non-commercial licence!) or espeak-ng. For monetised videos use Kokoro/Chatterbox languages or your own voice-over in the *Image → drawing video* tab.
* **Image prompts must be English** (the AI image models expect it) - the LLM writes them; edit in the table if needed.
* **Negative prompt** works with SD1.5 / SDXL (not FLUX).
* **Reference image/video:** with SD1.5 / SDXL an IP-Adapter copies the look of your reference; with other models only its colour palette is used.
* **Own hand:** choose *Custom upload*, upload a transparent PNG and set where the pen tip is (0-1 of width/height).
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="public gradio.live link (use on Colab)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    build().queue().launch(server_name=a.host, server_port=a.port, share=a.share, inbrowser=not a.share)
