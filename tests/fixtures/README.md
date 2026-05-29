# Parity test golden fixtures

These JSON files are the **golden** beat/downbeat times produced by the original
Python reference [`beat_this`](https://github.com/CPJKU/beat_this) on the shared
audio file `test_files/It Don't Mean A Thing - Kings of Swing.mp3`. The Rust parity
test (`tests/python_parity.rs`) compares the full Rust pipeline against them with the
standard ±70 ms MIR F-measure.

| File                | Model    | Checkpoint           | Used by                              |
| ------------------- | -------- | -------------------- | ------------------------------------ |
| `golden_small.json` | small    | `models/small1.ckpt` | `test_python_parity_small_model` (always-on) |
| `golden_full.json`  | standard | `models/final0.ckpt` | `test_python_parity_full_model` (gated on `models/beat_this.onnx`) |

`models/beat_this_small.onnx` is byte-identical (sha256) to the ONNX exported from
`small1.ckpt`, so the small golden matches the committed model exactly. The full
golden matches the FP32 `beat_this.onnx` fetched by `scripts/download-models.sh`.

Each file contains `beats`, `downbeats` (times in seconds) and a `provenance` block:

```json
{
  "beats": [...],
  "downbeats": [...],
  "provenance": {
    "checkpoint": "models/small1.ckpt",
    "audio": "It Don't Mean A Thing - Kings of Swing.mp3",
    "beat_this_version": "1.1.0",
    "beat_this_commit": "...",
    "postprocessing": "minimal",
    "fps": 50,
    "command": "uv run scripts/gen_golden.py models/small1.ckpt tests/fixtures/golden_small.json"
  }
}
```

Generated with `beat_this` **v1.1.0** (commit `b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c`,
the `refs/beat_this` pin), `minimal` post-processing, 50 fps. (`beat_this_commit`
reads `unknown` when the package is installed from the source zip rather than a git
checkout — the version is the reliable identifier.)

## Regenerating

Maintainer-only; needs `uv`. Regenerate if the checkpoint or the mel/inference graph
changes. The checkpoints are git-ignored; download them (or use the `beat_this`
shortnames `small1` / `final0`, which fetch from the cloud space) first.

```bash
cd beat-this-rs
uv run scripts/gen_golden.py models/small1.ckpt tests/fixtures/golden_small.json
uv run scripts/gen_golden.py models/final0.ckpt tests/fixtures/golden_full.json
```
