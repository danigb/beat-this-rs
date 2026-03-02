# Task 010: Model Management — Summary

Script to download Beat This! checkpoints and convert them to ONNX format for use with the Rust inference runtime.

## What was done

### Script: `scripts/ckpt2onnx.py`

A self-contained Python script using uv inline metadata (PEP 723) for dependency management. No manual environment setup needed — `uv run` handles everything.

Dependencies (installed automatically by uv):

- `torch>=2.0` — model loading and ONNX export
- `onnx`, `onnxscript` — ONNX format support
- `beat-this` — installed from GitHub (not from local references)

### How it works

1. Downloads the `.ckpt` file from the JKU cloud space into `models/` (skips if already present)
2. Loads the PyTorch Lightning checkpoint and extracts model hyperparameters
3. Instantiates a vanilla `BeatThis` model, strips the `model.` prefix from state dict keys
4. Wraps the model to return `(beat, downbeat)` tuple instead of a dict (ONNX requirement)
5. Exports to ONNX with dynamic batch and time axes (opset 17, legacy TorchScript exporter)

### ONNX model I/O

- **Input**: `spectrogram` — shape `(batch, time, 128)` — log-mel spectrogram, 128 bins, 50 FPS
- **Output**: `beat` — shape `(batch, time)` — beat logits
- **Output**: `downbeat` — shape `(batch, time)` — downbeat logits

Both outputs are raw logits (apply sigmoid for probabilities).

## Usage

```bash
uv run scripts/ckpt2onnx.py final0    # ~78 MB — main model
uv run scripts/ckpt2onnx.py small0    # ~8 MB  — smaller model
```

Output goes to `models/<name>.onnx`.

## Available models

All models use the same architecture (except `small` which has `transformer_dim=128` instead of 512). The "0" suffix is the random seed — only one seed is needed for inference.

| Name     | Size  | Description                                     |
| -------- | ----- | ----------------------------------------------- |
| `final0` | 78 MB | Default model, trained on all data except GTZAN |
| `small0` | 8 MB  | Smaller variant of the above                    |
