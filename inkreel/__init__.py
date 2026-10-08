"""InkReel - script -> narrated whiteboard-animation video, fully local."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# Every model download (LLM, TTS, diffusion ...) lands in ./data so the folder is portable
# and can be deleted / moved / put on Google Drive as one unit. Must be set before HF imports.
os.environ.setdefault("HF_HOME", str(DATA / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")

__version__ = "1.0.0"
