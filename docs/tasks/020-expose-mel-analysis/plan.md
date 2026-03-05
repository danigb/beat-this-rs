# Development Plan: Expose Mel Spectrogram (Task 020)

## Overview

Replace `process_audio` / `process_file` with `analyze_audio` / `analyze_file` that return a
richer `BeatAnalysis` type including the mel spectrogram and raw logits. Also remove `BeatResult`
since `BeatAnalysis` supersedes it. Add a `--mel` CLI flag that writes `.mel.npy`.

---

## Step 1 — Replace pipeline methods with `analyze_*`

**File:** [src/lib.rs](../../src/lib.rs)

Remove `BeatResult` re-export and replace with `BeatAnalysis`. Remove `process_audio` and
`process_file`. Add:

```rust
pub struct BeatAnalysis {
    pub beats: Vec<f32>,
    pub downbeats: Vec<f32>,
    pub mel: Tensor,
    pub beat_logits: Vec<f32>,
    pub downbeat_logits: Vec<f32>,
}
```

```rust
pub fn analyze_audio(&mut self, samples: &[f32], sample_rate: u32) -> Result<BeatAnalysis> {
    let samples = if sample_rate != TARGET_SAMPLE_RATE {
        audio::resample(samples.to_vec(), sample_rate, TARGET_SAMPLE_RATE)?
    } else {
        samples.to_vec()
    };
    let mel = self.mel.process(&samples)?;
    let (beat_logits, downbeat_logits) = self.inference.process(&mel)?;
    let result = self.post.process(&beat_logits, &downbeat_logits)?;
    Ok(BeatAnalysis {
        beats: result.beats,
        downbeats: result.downbeats,
        mel,
        beat_logits,
        downbeat_logits,
    })
}

pub fn analyze_file(&mut self, path: &Path) -> Result<BeatAnalysis> {
    let audio = load_audio(path, TARGET_SAMPLE_RATE)?;
    self.analyze_audio(&audio.samples, audio.sample_rate)
}
```

Re-export `BeatAnalysis` at the top of the file. Remove the `BeatResult` re-export.

**Also remove `BeatResult` from `postprocessing.rs`** — the struct is no longer part of the public
API. `PostProcessor::process` can return `(Vec<f32>, Vec<f32>)` (beats, downbeats) or keep
`BeatResult` as a private implementation detail if that reads cleaner internally.

---

## Step 2 — Add serde support to `Tensor`

**File:** [src/runtime.rs](../../src/runtime.rs)

Add serde behind a feature flag so JSON output remains opt-in and doesn't add a mandatory
dependency for library users:

**Cargo.toml:**

```toml
[features]
serde = ["dep:serde"]

[dependencies]
serde = { version = "1", features = ["derive"], optional = true }
```

**src/runtime.rs:**

```rust
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}
```

The binary (CLI) already uses `serde_json`; enable the feature for it in Cargo.toml:

```toml
[[bin]]
# or in the workspace/package features for the binary target:
```

Or simpler: just enable `serde` unconditionally in `[features] default = ["serde"]` since the
binary already depends on serde_json and there is no library-only use case that needs to exclude it.

---

## Step 3 — Add `.mel.npy` writer to `output.rs`

**File:** [src/output.rs](../../src/output.rs)

Add a function that writes the mel tensor as a numpy `.npy` file (no external crate needed —
the format is a fixed 128-byte header + raw little-endian f32 data):

```rust
pub fn write_mel_npy(path: &Path, mel: &Tensor) -> Result<()> {
    use std::io::Write;

    // mel shape: [1, T, 128] — write as [T, 128] (drop batch dim)
    let t_frames = mel.shape[1];
    let n_mels = mel.shape[2];

    // Build numpy header describing a float32 array of shape (T, 128), C order
    let dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        t_frames, n_mels
    );
    // Header block is padded to a multiple of 64 bytes (npy v1.0 spec)
    let header_len = dict.len() + 1; // +1 for trailing newline
    let padding = (64 - (10 + header_len) % 64) % 64;
    let padded = format!("{}{}\n", dict, " ".repeat(padding));

    let mut f = std::fs::File::create(path)?;
    f.write_all(b"\x93NUMPY")?;              // magic
    f.write_all(&[1, 0])?;                  // version 1.0
    f.write_all(&(padded.len() as u16).to_le_bytes())?; // header len (LE u16)
    f.write_all(padded.as_bytes())?;

    // Write float data (skip batch dim: data is already [1*T*128] row-major)
    let data_start = mel.shape[0] * 0; // offset 0 (batch=0 is the only slice)
    for &v in &mel.data[data_start..data_start + t_frames * n_mels] {
        f.write_all(&v.to_le_bytes())?;
    }

    Ok(())
}
```

---

## Step 4 — Add `--mel` flag to the CLI

**File:** [src/main.rs](../../src/main.rs)

Add the flag to `Cli`:

```rust
/// Write mel spectrogram as numpy .npy file [=FILE]
#[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "")]
mel: Option<String>,
```

Add `mel` to `OutputFlags`:

```rust
struct OutputFlags {
    json: Option<String>,
    beats: Option<String>,
    click: Option<String>,
    mix: Option<String>,
    mel: Option<String>,
    overwrite: bool,
}
```

Update `write_outputs` to accept `BeatAnalysis` instead of `BeatResult`, and add the mel write:

```rust
fn write_outputs(
    input: &Path,
    analysis: &BeatAnalysis,
    flags: &OutputFlags,
) -> anyhow::Result<Vec<String>> {
    // ... existing json/beats/click/mix writes using &analysis.beats / &analysis.downbeats ...

    if let Some(path) = resolve_output_path(input, &flags.mel, "mel.npy") {
        if write_if_needed(&path, flags.overwrite, |p| {
            output::write_mel_npy(p, &analysis.mel)
        })? {
            written.push(path.display().to_string());
        }
    }

    Ok(written)
}
```

Update `process_single_file` to return `BeatAnalysis` instead of just `BeatResult`, and thread it
through `run_pipeline` and `run_batch`. The beat/downbeat fields are still available on
`BeatAnalysis` so all existing output paths remain unchanged.

---

## Step 5 — Tests

- Unit test `write_mel_npy`: write a small known tensor, read back the bytes, verify magic, shape
  in header, and a few float values.
- Optionally: a round-trip test loading the `.npy` from Python in CI (out of scope for this PR).

---

## File Change Summary

| File                    | Change                                                                                                                  |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `src/lib.rs`            | Add `BeatAnalysis`; add `analyze_audio`, `analyze_file`; remove `process_audio`, `process_file`, `BeatResult` re-export |
| `src/postprocessing.rs` | Remove or demote `BeatResult` to private implementation detail                                                          |
| `src/runtime.rs`        | Add `#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]` to `Tensor`                                        |
| `src/output.rs`         | Update all functions to accept `BeatAnalysis`; add `write_mel_npy`                                                      |
| `src/main.rs`           | Add `--mel` flag; update `OutputFlags`, `write_outputs`, `process_single_file` to use `BeatAnalysis`                    |
| `Cargo.toml`            | Add optional serde dep; consider enabling by default                                                                    |
