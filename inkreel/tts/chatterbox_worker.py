"""Runs inside .venv_chatterbox. Protocol: one JSON job per stdin line -> one JSON reply per stdout line."""
import json
import sys

proto = sys.stdout            # keep the real stdout for the protocol, send library chatter to stderr
sys.stdout = sys.stderr
device = sys.argv[1] if len(sys.argv) > 1 else "cpu"

import torch  # noqa: E402
import soundfile as sf  # noqa: E402

if device == "cpu":
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "map_location": torch.device("cpu")})

try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS as Model
    multilingual = True
except Exception:
    from chatterbox.tts import ChatterboxTTS as Model
    multilingual = False

model = Model.from_pretrained(device=device)
print(json.dumps({"ready": True, "multilingual": multilingual}), file=proto, flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        job = json.loads(line)
        kw = {}
        if job.get("ref"):
            kw["audio_prompt_path"] = job["ref"]
        if multilingual:
            kw["language_id"] = job.get("lang") or "en"
        wav = model.generate(job["text"], **kw)
        sf.write(job["out"], wav.squeeze().cpu().numpy(), model.sr)
        print(json.dumps({"ok": True}), file=proto, flush=True)
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}), file=proto, flush=True)
