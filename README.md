# Sermon Broadcaster

A Flask web app that converts church service videos into broadcast-ready radio audio. Automatically identifies the sermon within a full service, fits it to a target broadcast duration, and optionally wraps it with intro and outro bumpers (with either an AI-selected dynamic teaser or a pre-mixed stock teaser).

Built for [Grace Free Lutheran Church](https://gracefreelutheranchurch.org/) to streamline weekly sermon preparation for local radio broadcast.

## Features

- **Two input methods:** YouTube URL or direct video/audio file upload (up to 2 GB)
- **Intelligent sermon boundary detection** using Claude API with word-level timestamp refinement
  - Snaps end boundary to "Amen" before liturgical transitions ("let us stand," "stand and sing")
  - Handles both with-prayer and without-prayer endpoints, picks whichever fits best
  - Snaps start boundary by matching the first spoken sermon words
- **Duration fitting** to any target length:
  - Trims long pauses (configurable threshold)
  - Expands short pauses if sermon needs to be longer
  - Tempo adjustment (±7%) without pitch change
- **Two bumper variants** (selectable independently or together):
  - **Dynamic teaser:** Claude selects a compelling 13-20 second clip from the sermon, extracted and mixed into the intro
  - **Stock teaser:** Pre-mixed evergreen intro file used as-is
  - Auto-fallback: if dynamic teaser fails, automatically produces stock variant
- **Cloud transcription** via OpenAI Whisper API (Opus encoding for timestamp accuracy)
- **Production deployment** as a systemd service with automatic cleanup

## Architecture

```
┌─────────────┐
│  Flask App  │  app.py — UI, file uploads, job tracking
└──────┬──────┘
       │
┌──────▼──────────┐
│   Orchestrator  │  pipeline/orchestrator.py — coordinates all stages
└──────┬──────────┘
       │
       ├─ downloader.py        (YouTube via yt-dlp, or local file conversion)
       ├─ transcriber_cloud.py (OpenAI Whisper API)
       ├─ boundary_detector.py (Claude API + word-level refinement)
       ├─ audio_processor.py   (silence detection, trim/expand, tempo)
       ├─ teaser_selector.py   (Claude API + verbatim text matching)
       └─ assembler.py         (intro + sermon + outro concatenation)
```

## Pipeline Stages

1. **Download/Convert** — YouTube via yt-dlp, or ffmpeg conversion for uploads
2. **Transcribe** — OpenAI Whisper API with word-level timestamps (Opus compression for alignment)
3. **Detect boundaries** — Claude identifies sermon start/end, then word-level refinement snaps to exact positions
4. **Extract sermon segment** — Direct sample-accurate extraction from raw audio
5. **Fit to duration** — Silence trim/expand + tempo adjustment to hit target length
6. **Assemble output(s)** — Sermon-only, dynamic-teaser variant, and/or stock-teaser variant

## Setup

### Prerequisites

- Python 3.10+
- ffmpeg with libopus, libmp3lame, librubberband
- yt-dlp (system or pip-installed)

### Installation

```bash
git clone https://github.com/<your-username>/sermon-broadcaster.git
cd sermon-broadcaster

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env with API keys
cat > .env << EOF
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
TRANSCRIBER=cloud
EOF

# Add your bumper files to assets/
# - assets/intro.mp3      (intro music with pre-ducked teaser window)
# - assets/outro.mp3      (outro bumper)
# - assets/intro_stock.mp3 (optional: pre-mixed stock teaser intro)
```

### Run locally

```bash
python app.py
# Browse to http://localhost:5003
```

### Production deployment (systemd)

```bash
sudo cp sermon-broadcaster.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sermon-broadcaster
sudo systemctl start sermon-broadcaster
```

### Periodic cleanup (cron)

```cron
# Weekly cleanup of old files
0 3 * * 0 find /opt/sermon-broadcaster/pipeline/cache -type f -mtime +30 -delete
0 3 * * 0 find /opt/sermon-broadcaster/output -type f -mtime +60 -delete
0 3 * * 0 find /opt/sermon-broadcaster/uploads -type f -mtime +7 -delete
```

## Configuration

All tunable parameters live in `config.py`:

| Setting | Default | Purpose |
|---|---|---|
| `DEFAULT_TARGET_DURATION` | `27:18` | Sermon-only target |
| `DEFAULT_BROADCAST_DURATION` | `29:30` | Full broadcast target (with bumpers) |
| `TEASER_WINDOW_START` | `12.0` | Where teaser sits in the intro (seconds) |
| `TEASER_WINDOW_END` | `35.0` | End of teaser window |
| `MAX_SPEEDUP` | `1.08` | Maximum tempo speedup (8%) |
| `MAX_SLOWDOWN` | `0.93` | Maximum tempo slowdown (7%) |
| `MAX_PAUSE_DURATION_MS` | `1500` | Trim pauses longer than this |
| `OUTPUT_BITRATE` | `128k` | Final MP3 bitrate |

## Output

Files are written to `output/` with timestamps:
- `sermon_YYYYMMDD_HHMMSS.mp3` — sermon only (no bumpers)
- `sermon_YYYYMMDD_HHMMSS_dynamic.mp3` — with intro (AI teaser) + outro
- `sermon_YYYYMMDD_HHMMSS_stock.mp3` — with intro (stock teaser) + outro

## Built With

- [Flask](https://flask.palletsprojects.com/) — web framework
- [Anthropic Claude API](https://docs.anthropic.com/) — sermon analysis & teaser selection
- [OpenAI Whisper API](https://platform.openai.com/docs/guides/speech-to-text) — transcription
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube extraction
- [ffmpeg](https://ffmpeg.org/) — audio processing
- [soundfile](https://python-soundfile.readthedocs.io/) / [numpy](https://numpy.org/) — sample-accurate manipulation

## License

Personal/ministry project. Use at your own discretion.
