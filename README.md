# Sermon Broadcaster

A Flask web app that converts church service videos into broadcast-ready radio audio. Automatically identifies the sermon within a full service, fits it to a target broadcast duration, and optionally wraps it with intro and outro bumpers (with either an AI-selected dynamic teaser or a pre-mixed stock teaser).

Built for [Grace Free Lutheran Church](https://gracefree.com/) to streamline weekly sermon preparation for local radio broadcast.

## Features

- **Two input methods:** YouTube URL or direct video/audio file upload (5 GB by default)
- **Human review editor** with a waveform, clickable transcript, and exact sermon/teaser markers
- **Two-stage workflow:** analyze once, review selections, then render without downloading or transcribing again
- **Intelligent sermon boundary detection** using Claude API with word-level timestamp refinement
  - Snaps end boundary to "Amen" before liturgical transitions ("let us stand," "stand and sing")
  - Handles both with-prayer and without-prayer endpoints, picks whichever fits best
  - Snaps start boundary by matching the first spoken sermon words
- **Duration fitting** to any target length:
  - Trims long pauses (configurable threshold)
  - Expands short pauses if sermon needs to be longer
  - Tempo adjustment (up to 8% faster or 7% slower) without pitch change
- **Two bumper variants** (selectable independently or together):
  - **Dynamic teaser:** Claude selects a compelling 12-22 second clip from the sermon, extracted and mixed into the intro
  - **Stock teaser:** Pre-mixed evergreen intro file used as-is
  - Stock mode is enabled only when `assets/intro_stock.mp3` is installed
- **Selectable transcription:** OpenAI Whisper API, a local HTTP service, or faster-whisper
- **Production deployment** with single-worker Gunicorn and systemd

## Architecture

```
┌─────────────┐
│  Flask App  │  app.py — UI, file uploads, job tracking
└──────┬──────┘
       │
┌──────▼──────────┐
│ Review Workflow │  pipeline/review_workflow.py — analysis, review, render
└──────┬──────────┘
       │
       ├─ downloader.py        (YouTube via yt-dlp, or local file conversion)
       ├─ transcription.py     (OpenAI, local HTTP, or faster-whisper)
       ├─ boundary_detector.py (Claude API + word-level refinement)
       ├─ audio_processor.py   (silence detection, trim/expand, tempo)
       ├─ teaser_selector.py   (Claude API + verbatim text matching)
       └─ assembler.py         (intro + sermon + outro concatenation)
```

## Review Workflow

1. **Analyze** — download/convert, transcribe, build the waveform, and generate optional AI suggestions
2. **Review** — confirm sermon start/end and teaser start/end in the browser
3. **Preflight** — show the selected duration, available sermon time, warnings, and blockers
4. **Render** — extract the confirmed sermon, fit it conservatively, mix the teaser, and assemble output(s)
5. **Verify** — reject unsafe stretching or a result that cannot meet the required duration

Analysis artifacts are stored under `state/review_jobs/<job_id>/` so a review can be resumed from Job History.

## Setup

### Prerequisites

- Python 3.11+
- ffmpeg/ffprobe with libopus and libmp3lame
- The Python dependencies from `requirements.txt` (including yt-dlp)

### Installation

```bash
git clone https://github.com/larryherzogjr/sermon-broadcaster.git
cd sermon-broadcaster

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env with API keys and backend configuration
cp .env.example .env
# Edit .env before starting the app.

# Add your bumper files to assets/
# - assets/intro.mp3      (intro music with pre-ducked teaser window)
# - assets/outro.mp3      (outro bumper)
# - assets/intro_stock.mp3 (optional: pre-mixed stock teaser intro)
```

When both bumper variants are requested, their intro-plus-outro durations must
match so both finished files can satisfy the same broadcast target.

### Run locally

```bash
python app.py
# Browse to http://localhost:5003
```

The app performs processing preflight before accepting a job. Missing binaries,
API credentials, or selected bumper files are reported immediately. Runtime
readiness is also available at `GET /api/health`.

### Validation

```bash
pip install -r requirements-dev.txt
make check
```

`make check` compiles the Python sources and runs the pytest suite. The same
command runs in GitHub Actions.

### Production deployment (systemd)

Review `User`, `WorkingDirectory`, and `EnvironmentFile` in
`sermon-broadcaster.service` before installing it; the checked-in values target
the current `/opt/sermon-broadcaster` deployment.

```bash
sudo cp sermon-broadcaster.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sermon-broadcaster
sudo systemctl start sermon-broadcaster
```

The supplied unit runs one Gunicorn worker with multiple request threads. Keep
it at one worker: processing currently uses in-process background threads and
SQLite job reconciliation. Place an authenticating reverse proxy in front of
the app before exposing it outside a trusted private network.

### Periodic cleanup (cron)

```cron
# Weekly cleanup of old files
0 3 * * 0 find /opt/sermon-broadcaster/pipeline/cache -type f -mtime +30 -delete
0 3 * * 0 find /opt/sermon-broadcaster/output -type f -mtime +60 -delete
0 3 * * 0 find /opt/sermon-broadcaster/uploads -type f -mtime +7 -delete
0 3 * * 0 find /opt/sermon-broadcaster/state/review_jobs -mindepth 1 -maxdepth 1 -type d -mtime +30 -exec rm -rf {} +
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
| `MAX_UPLOAD_GB` | `5` | Maximum request/upload size |
| `APP_HOST` / `APP_PORT` | `0.0.0.0` / `5003` | Local server bind address |

## Output

Files are written to `output/` using the timestamp-based job ID:

- `sermon_<job_id>.mp3` — sermon only (no bumpers)
- `sermon_<job_id>_dynamic.mp3` — with intro (AI teaser) + outro
- `sermon_<job_id>_stock.mp3` — with intro (stock teaser) + outro

## Built With

- [Flask](https://flask.palletsprojects.com/) — web framework
- [Anthropic Claude API](https://docs.anthropic.com/) — sermon analysis & teaser selection
- [OpenAI Whisper API](https://platform.openai.com/docs/guides/speech-to-text) — transcription
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube extraction
- [ffmpeg](https://ffmpeg.org/) — audio processing
- [soundfile](https://python-soundfile.readthedocs.io/) / [numpy](https://numpy.org/) — sample-accurate manipulation

## License

Personal/ministry project. Use at your own discretion.
