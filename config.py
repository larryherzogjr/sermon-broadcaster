import os
from dotenv import load_dotenv

load_dotenv()

# Claude API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Per-task model selection — different tasks have different needs.
# Boundary detection is mechanical (find start/end), Sonnet handles fine.
# Teaser selection requires more "taste" — Opus can pick more compelling clips
# and copy text more reliably verbatim, reducing retries.
CLAUDE_MODEL_BOUNDARY = os.getenv("CLAUDE_MODEL_BOUNDARY", "claude-sonnet-4-20250514")
CLAUDE_MODEL_TEASER = os.getenv("CLAUDE_MODEL_TEASER", "claude-opus-4-7")

# Legacy fallback — if old code references CLAUDE_MODEL, use boundary model
CLAUDE_MODEL = CLAUDE_MODEL_BOUNDARY

# Transcription: "local" for faster-whisper on CPU, "cloud" for OpenAI Whisper API
TRANSCRIBER = os.getenv("TRANSCRIBER", "cloud")

# Local Whisper settings (only used when TRANSCRIBER=local)
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"  # int8 is fastest on CPU

# OpenAI API (only used when TRANSCRIBER=cloud)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Audio defaults
DEFAULT_TARGET_DURATION = "27:18"          # Sermon-only target (no bumpers)
DEFAULT_BROADCAST_DURATION = "29:30"       # Full broadcast target (with bumpers)
OUTPUT_FORMAT = "mp3"
OUTPUT_BITRATE = "128k"
OUTPUT_SAMPLE_RATE = 44100

# Bumper paths
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
INTRO_PATH = os.path.join(ASSETS_DIR, "intro.mp3")
OUTRO_PATH = os.path.join(ASSETS_DIR, "outro.mp3")

# Teaser window in the intro (seconds)
TEASER_WINDOW_START = 12.0
TEASER_WINDOW_END = 35.0

# Silence manipulation thresholds
MAX_PAUSE_DURATION_MS = 1500      # Trim pauses longer than this (ms)
TARGET_PAUSE_DURATION_MS = 800    # Trim them down to this (ms)
MIN_PAUSE_FOR_INSERT_MS = 300     # Minimum existing pause to expand when adding time
MAX_PAUSE_INSERT_MS = 2000        # Don't expand any single pause beyond this

# Tempo adjustment limits
MAX_SPEEDUP = 1.08   # Don't speed up more than 8%
MAX_SLOWDOWN = 0.93  # Don't slow down more than 7%

# Paths
WORK_DIR = os.path.join(os.path.dirname(__file__), "work")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
