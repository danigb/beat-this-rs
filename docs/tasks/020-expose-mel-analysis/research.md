# Research: Expose Mel Spectrogram (Task 020)

## Codebase Summary

The pipeline in [src/lib.rs](../../src/lib.rs) already computes mel as a local variable inside
`process_audio()` and immediately discards it:

```rust
let mel = self.mel.process(&samples)?;         // Tensor { shape: [1, T, 128] }
let (beat_logits, downbeat_logits) = self.inference.process(&mel)?;
self.post.process(&beat_logits, &downbeat_logits)  // mel is dropped here
```

The change is minimal: thread `mel` into the return value.

## Proposed Implementation

### New type in `lib.rs` (or a new `analysis.rs`)

```rust
pub struct BeatAnalysis {
    pub beats: Vec<f32>,      // beat times in seconds
    pub downbeats: Vec<f32>,  // downbeat times in seconds
    pub mel: Tensor,          // shape [1, T, 128] at 50 fps
}
```

### New methods alongside existing ones

```rust
impl<S: InferenceSession> BeatThis<S> {
    pub fn analyze_audio(&mut self, samples: &[f32], sample_rate: u32) -> Result<BeatAnalysis> {
        let samples = if sample_rate != TARGET_SAMPLE_RATE { ... } else { ... };
        let mel = self.mel.process(&samples)?;
        let (beat_logits, downbeat_logits) = self.inference.process(&mel)?;
        let BeatResult { beats, downbeats } = self.post.process(&beat_logits, &downbeat_logits)?;
        Ok(BeatAnalysis { beats, downbeats, mel })
    }

    pub fn analyze_file(&mut self, path: &Path) -> Result<BeatAnalysis> {
        let audio = load_audio(path, TARGET_SAMPLE_RATE)?;
        self.analyze_audio(&audio.samples, audio.sample_rate)
    }
}
```

Keep `process_audio` and `process_file` as thin wrappers over the new methods
(extract `BeatResult` from `BeatAnalysis`) to maintain backward compatibility.

### CLI: `--mel` flag

The task says: don't change existing JSON output, add `--mel` that writes a `.mel.json`.

**Concern: file size.** A 3-minute song ≈ 9000 frames × 128 bands = 1.15M floats.
As JSON that is ~7 MB per file. Consider:

- `--mel` → writes `.mel.npy` (raw f32 array, numpy-compatible binary, ~4.4 MB, much faster to read)
- `--mel-json` → writes `.mel.json` if JSON is required for interop

The npy format is trivial to write (80-byte header + raw f32 bytes) and readable by numpy/Python
without any special library on the consumer side.

**Alternative**: write a flat binary `.mel.bin` with a small header (shape as u32 triplet, then f32
data). Even simpler than npy but less standard.

**Recommendation**: implement `.mel.npy` as the default for `--mel`; it's the most useful format
for the ondas/rosa use case since Python reads it with `np.load`.

## Integration Path for ondas

With `analyze_file()` available:
1. ondas calls `bt.analyze_file(path)` → gets `BeatAnalysis { beats, downbeats, mel }`
2. MFCC = DCT applied to `analysis.mel.data` (reshaped as `[T, 128]`) — no rosa needed for mel
3. Audio is loaded only once (beat-this-rs handles resampling to 22050 Hz)

The mel from the ONNX model is already log-scaled and normalized to match the training pipeline,
so it should be suitable as a drop-in for MFCC computation.

## Additional Suggestions

### 1. Expose raw logits (optional, low cost)

`BeatAnalysis` could optionally include:
```rust
pub beat_logits: Vec<f32>,
pub downbeat_logits: Vec<f32>,
```
These are already computed before being discarded. They allow downstream callers to apply their
own peak-picking thresholds or re-use the per-frame probabilities (e.g. for visualization).
This costs nothing since the data already exists.

### 2. `Tensor` serialization with serde

`Tensor` in [src/runtime.rs](../../src/runtime.rs) has no `serde` support. Adding
`#[derive(Serialize, Deserialize)]` behind a `serde` feature flag would make mel JSON output
trivial and also enable future serialization of logits. The `output` module already uses
`serde_json`, so the dependency is already present.

### 3. Refactor `process_audio` as a wrapper

Once `analyze_audio` exists, `process_audio` can become:
```rust
pub fn process_audio(&mut self, samples: &[f32], sample_rate: u32) -> Result<BeatResult> {
    let a = self.analyze_audio(samples, sample_rate)?;
    Ok(BeatResult { beats: a.beats, downbeats: a.downbeats })
}
```
This removes duplicated pipeline code.

### 4. Consider a 2D mel shape

The mel shape is `[1, T, 128]` — the batch dim is always 1 and is an artifact of the ONNX model.
Downstream callers must remember to index with `[0]`. Wrapping in a `MelSpectrogram` newtype with
`fn frames(&self) -> usize` and `fn frame(&self, t: usize) -> &[f32]` accessors would be nicer
ergonomically, but is optional since the raw `Tensor` is sufficient for the immediate use case.

## Implementation Order

1. Add `BeatAnalysis` struct to `lib.rs`
2. Add `analyze_audio` / `analyze_file` methods; refactor `process_audio`/`process_file` as wrappers
3. Re-export `BeatAnalysis` from `lib.rs` alongside `BeatResult`
4. Add `--mel` CLI flag in `main.rs` with an output writer in `output.rs`
5. (Optional) Add serde support to `Tensor`
