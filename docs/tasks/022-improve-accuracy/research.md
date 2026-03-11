# Research: Audio Chunk Padding Fix

## Problem

The Rust implementation was zero-padding **all** chunks to full `CHUNK_SIZE` (1500 frames), regardless of whether the audio was short or long. This caused the model to receive mostly-zero input for short audio files, degrading prediction accuracy.

## Fix Applied (branch: `feat/audio-chunk-padding`)

**File changed**: `src/inference.rs` (single file, +83/-25 lines)

### What changed in `extract_chunk`

Previously, `extract_chunk` always created a buffer of `CHUNK_SIZE * n_mels` zeros and filled in the actual audio data. Every chunk was exactly 1500 frames regardless of audio length.

Now, `extract_chunk` receives the `full_time` parameter and applies **conditional padding**:

- **Short audio** (`full_time <= STRIDE`, i.e., fits in a single chunk): Right padding is capped at `BORDER_SIZE` (6 frames). A 100-frame audio becomes `6 + 100 + 6 = 112` frames, not 1500.
- **Long audio** (multiple chunks): Padding still fills to `CHUNK_SIZE` (1500), preserving existing behavior.

### What changed in the prediction loop

- `chunk_time` is now derived from the actual chunk shape instead of being hardcoded to `CHUNK_SIZE`.
- Border stripping uses `chunk_time - BORDER_SIZE` instead of `CHUNK_SIZE - BORDER_SIZE`.
- The write loop uses `zip` iterators instead of index-based access, making it safe against variable chunk sizes.

## Comparison with Python Reference

**File**: `references/beat_this/beat_this/inference.py`, line 133

```python
right=max(0, min(border_size, start + chunk_size - len(spect))),
```

The Python `split_piece` function caps right padding at `border_size` via `min(border_size, ...)`. This means:

- For short audio (single chunk): right padding is at most `border_size` (6 frames) — the chunk is NOT padded to full `chunk_size`.
- For long audio: chunks that don't extend beyond the spectrogram end get `right=0`; only the last chunk gets up to `border_size` right padding.

**The Rust fix now matches this Python behavior exactly.**

| Scenario | Python | Rust (before) | Rust (after) |
|----------|--------|---------------|--------------|
| 100-frame audio | 112 frames (6+100+6) | 1500 frames | 112 frames (6+100+6) |
| 5000-frame audio, first chunk | 1500 frames | 1500 frames | 1500 frames |
| 5000-frame audio, middle chunk | 1500 frames | 1500 frames | 1500 frames |

## Comparison with C++ Reference

**File**: `references/beat_this_cpp/Source/InferenceProcessor.cpp`, line 98

```cpp
int right_pad = std::max(0, std::min(border_size, start + chunk_size - len_spect));
```

The C++ implementation is a **direct port of the Python logic** — it uses the same `max(0, min(border_size, ...))` formula. The C++ `split_piece` and `zeropad` functions mirror the Python ones almost line-for-line.

**All three implementations (Python, C++, Rust) now agree on padding behavior.**

## Structural Divergence: Conditional vs Unified Formula

Although the **results are equivalent**, the Rust code uses a different structure than Python/C++.

### Python & C++ — Unified formula for all chunks

```python
# Python (inference.py:133) — same formula always, for all chunks
right=max(0, min(border_size, start + chunk_size - len(spect)))
```

```cpp
// C++ (InferenceProcessor.cpp:98) — direct translation of Python
int right_pad = std::max(0, std::min(border_size, start + chunk_size - len_spect));
```

This single expression handles every case naturally:
- Middle chunks (no overflow): `start + chunk_size - len(spect)` is negative → `max(0, ...)` → 0
- Last chunk (slight overflow): result is between 0 and `border_size`
- Short audio (large overflow): `min(border_size, ...)` caps it at 6

### Rust — Conditional branching

```rust
// Rust (inference.rs:144-149)
let pad_right = if full_time <= STRIDE {
    BORDER_SIZE.min((start + CHUNK_SIZE as i32) as usize - full_time)
} else {
    CHUNK_SIZE - pad_left - n_frames
};
```

Two separate strategies based on audio length:
- **Short audio branch**: Caps at `BORDER_SIZE` (matches Python's `min(border_size, ...)`)
- **Long audio branch**: Fills to `CHUNK_SIZE` (equivalent to Python for adjusted-last-chunk, but by a different path)

### Why it still works

For long audio, `avoid_short_end` ensures the last chunk's start is adjusted so `start + CHUNK_SIZE - full_time` equals exactly `BORDER_SIZE`. So `CHUNK_SIZE - pad_left - n_frames` always equals `min(BORDER_SIZE, start + CHUNK_SIZE - full_time)` for the adjusted last chunk.

### Potential simplification

The Rust `extract_chunk` could be simplified to mirror Python/C++ exactly with a single formula:

```rust
let pad_right = 0.max((start + CHUNK_SIZE as i32 - full_time as i32).min(BORDER_SIZE as i32)) as usize;
```

This would:
- Remove the `full_time` parameter from `extract_chunk` (it could be derived from `mel.shape[1]` again)
- Eliminate the conditional branching
- Make the code a direct translation of Python/C++, easier to verify correctness

## Full Comparison

| Aspect | Python | C++ | Rust (current) |
|--------|--------|-----|----------------|
| Right padding formula | `max(0, min(border_size, ...))` | `max(0, min(border_size, ...))` | Conditional branch (equivalent result) |
| `avoid_short_end` condition | `len > chunk_size - 2*border_size` | same | `full_time > STRIDE` (equivalent) |
| Start generation | `np.arange(...)` | `for` loop | `while` loop (equivalent) |
| Overlap mode | "keep_first" (reverse iteration) | Reverse iteration | Reverse iteration |
| Border stripping | `pred[border_size:-border_size]` | Loop with bounds check | Slice + zip (equivalent) |
| Write destination | Slice assignment (PyTorch clamps) | Loop with bounds check | Loop with bounds check |

**All three produce identical outputs.** The only divergence is structural: the Rust version uses conditional logic where Python/C++ use a unified formula. This could be simplified to match them line-for-line.
