#!/usr/bin/env bash
# Download and convert Beat This! models into models/
# Requires: Python 3.10+ and uv (https://docs.astral.sh/uv/)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Downloading and converting standard model (~83 MB)..."
uv run scripts/ckpt2onnx.py final0
mv models/final0.onnx models/beat_this.onnx
echo "Saved models/beat_this.onnx"

echo ""
echo "Optional: download the small model (~10 MB):"
echo "  uv run scripts/ckpt2onnx.py small0"
echo "  mv models/small0.onnx models/beat_this_small.onnx"
