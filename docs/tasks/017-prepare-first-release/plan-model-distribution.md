# Plan: Model Distribution for v0.1.0 Release

## Context

The project needs two ONNX models to run: `mel_spectrogram.onnx` (~270 KB) and `beat_this.onnx` (~83 MB). Currently the entire `models/` directory is git-ignored, so users have no way to obtain models from the repo alone. For the release:

- **Mel model**: Small enough to commit to git. Will be included in the repo.
- **Beat model(s)**: Too large for git. Users will download and convert via the existing `scripts/ckpt2onnx.py`.

## Changes

### 1. Commit mel model into `models/mel_spectrogram.onnx`

Un-ignore just this file by adding a negation rule to `.gitignore`:

```
models/
!models/mel_spectrogram.onnx
```

This keeps the current default path (`models/mel_spectrogram.onnx`) working with zero code changes. The CLI, lib API, Dockerfile, and tests all already reference this path.

### 2. Add a `scripts/download-models.sh` script

A simple shell script that wraps `ckpt2onnx.py` with sensible defaults:

```bash
#!/usr/bin/env bash
# Downloads and converts Beat This! models into models/
# Requires: Python 3.10+, uv
uv run scripts/ckpt2onnx.py final0
# Rename to match CLI default
mv models/final0.onnx models/beat_this.onnx
echo "Optional: download small model"
echo "  uv run scripts/ckpt2onnx.py small0"
echo "  mv models/small0.onnx models/beat_this_small.onnx"
```

### 3. Update README model setup section

Replace the current "place files in models/" text with concrete instructions:

```markdown
## Model Setup

The mel spectrogram model is included in the repository. To download and convert
the beat tracking model:

    ./scripts/download-models.sh

This requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

Alternatively, download and convert manually:

    uv run scripts/ckpt2onnx.py final0
    mv models/final0.onnx models/beat_this.onnx

For the small model (~10 MB):

    uv run scripts/ckpt2onnx.py small0
    mv models/small0.onnx models/beat_this_small.onnx
```

## Files to modify

- `.gitignore` — add `!models/mel_spectrogram.onnx` negation
- `scripts/download-models.sh` — new file (convenience wrapper)
- `README.md` — update model setup instructions

## Verification

1. `git add models/mel_spectrogram.onnx` succeeds (not ignored)
2. `git status` still ignores other files in `models/`
3. `./scripts/download-models.sh` downloads and converts the standard model
4. `cargo run -- --help` works (mel model available, beat model shows clear error if not downloaded)
