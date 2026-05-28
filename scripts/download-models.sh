#!/usr/bin/env bash
# Download the Beat This! FP32 beat model from GitHub Releases into models/.
# Requires: curl and a SHA-256 tool (sha256sum on Linux, shasum on macOS).
# No Python/torch/uv — the small model used by the test suite is committed to git.
set -euo pipefail

REPO="danigb/beat-this-rs"
ASSET="beat_this.onnx"
# The FP32 model lives in a dedicated release decoupled from code/version releases.
RELEASE_TAG="model-large"
BASE="https://github.com/${REPO}/releases/download/${RELEASE_TAG}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p models

echo "Downloading ${ASSET} from the ${RELEASE_TAG} release of ${REPO}..."
curl -fL --retry 3 -o "models/${ASSET}" "${BASE}/${ASSET}"
curl -fL --retry 3 -o "models/${ASSET}.sha256" "${BASE}/${ASSET}.sha256"

echo "Verifying SHA-256..."
(
    cd models
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -c "${ASSET}.sha256"
    else
        shasum -a 256 -c "${ASSET}.sha256"
    fi
)
rm -f "models/${ASSET}.sha256"
echo "Saved models/${ASSET}"
