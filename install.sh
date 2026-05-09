#!/bin/bash
# One-shot installer for semantic 3D reconstruction pipeline
# Run once before using run.py
#
# Requirements:
#   - conda (miniforge/miniconda)
#   - CUDA 12.4
#   - gcc-11

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
MAST3R_DIR="$REPO_DIR/extern/mast3r"

echo "==> Setting up Semantic 3D Reconstruction Pipeline"
echo "    Repo: $REPO_DIR"

# Ensure pip is available
echo ""
echo "==> Ensuring pip is available..."
conda install pip -y -q 2>/dev/null || true
python -m pip install --upgrade pip -q

# MASt3R
if [ ! -d "$MAST3R_DIR" ]; then
    echo ""
    echo "==> Cloning MASt3R..."
    mkdir -p "$REPO_DIR/extern"
    git clone --recursive https://github.com/naver/mast3r.git "$MAST3R_DIR"
else
    echo ""
    echo "==> MASt3R already present, updating submodules..."
    cd "$MAST3R_DIR" && git submodule update --init --recursive && cd "$REPO_DIR"
fi

echo ""
echo "==> Installing MASt3R dependencies..."
python -m pip install -r "$MAST3R_DIR/requirements.txt" -q
python -m pip install -r "$MAST3R_DIR/dust3r/requirements.txt" -q

# Project dependencies
echo ""
echo "==> Installing project dependencies..."
python -m pip install -r "$REPO_DIR/requirements.txt" -q

# YOLO weights
echo ""
echo "==> Pre-downloading YOLO weights..."
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" 2>/dev/null || true

echo ""
echo "==> Installation complete."
echo ""
echo "Usage:"
echo "  python run.py --video path/to/video.mp4 --output output/"
echo "  python visualise.py --pointcloud output/reconstruction.ply --semantic output/semantic_map.json"
echo ""
echo "Note: MASt3R model weights (~1.5GB) download automatically on first run."
