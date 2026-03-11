# Summary: Audio Chunk Padding Fix

## Changes

Single file changed: `src/inference.rs`

### 1. `extract_chunk` — unified padding formula

Right padding now uses the same formula as Python and C++:

```rust
// max(0, min(border_size, start + chunk_size - len))
let pad_right =
    0.max((start + CHUNK_SIZE as i32 - full_time as i32).min(BORDER_SIZE as i32)) as usize;
```

This caps right padding at `BORDER_SIZE` (6 frames) for all chunks. For short audio, this avoids zero-padding to 1500 frames. For long audio, `avoid_short_end` ensures chunks naturally fill to `CHUNK_SIZE`.

The `full_time` parameter was removed from the function signature — it's derived from `mel.shape[1]`.

### 2. Prediction loop — dynamic chunk size

- `chunk_time` is read from the actual chunk shape, not hardcoded to `CHUNK_SIZE`.
- Border stripping uses `chunk_time - BORDER_SIZE`.
- Write loop uses `zip` iterators for safety with variable chunk sizes.

## Results

All three implementations (Python, C++, Rust) now produce identical outputs:

| Scenario | Before | After |
|----------|--------|-------|
| 100-frame audio | 1500-frame chunk (93% zeros) | 112-frame chunk (6+100+6) |
| 5000-frame audio | 1500-frame chunks | 1500-frame chunks (unchanged) |

The padding formula, start generation, overlap mode, border stripping, and aggregation all match Python and C++ exactly. No structural divergences remain.
