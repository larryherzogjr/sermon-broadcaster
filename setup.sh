#!/bin/bash
# Sermon Broadcaster - Setup Script
# Run this after installing system dependencies:
#   sudo apt update && sudo apt install -y ffmpeg python3.11-venv python3-pip git

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================"
echo " Sermon Broadcaster Setup"
echo "================================"
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "[1/4] Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "[1/4] Virtual environment already exists, skipping."
fi

# Activate
source venv/bin/activate

# Install Python dependencies
echo "[2/4] Installing Python dependencies (this takes a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

# Set up .env if not present
if [ ! -f ".env" ]; then
    echo "[3/4] Creating .env from template..."
    cp .env.example .env
    echo "     >>> IMPORTANT: Edit .env and configure API keys/backend settings <<<"
else
    echo "[3/4] .env already exists, skipping."
fi

# Create directories
echo "[4/4] Creating work directories..."
mkdir -p work output uploads state/review_jobs

echo ""
echo "================================"
echo " Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env and configure Anthropic plus your transcription backend"
echo "  2. Activate the environment:  source venv/bin/activate"
echo "  3. Start the app:            python app.py"
echo "  4. Open in browser:          http://<vm-ip>:5003"
echo ""
echo "For local CPU transcription, the Whisper model downloads on first use."
echo ""
