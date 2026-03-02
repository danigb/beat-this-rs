use anyhow::{anyhow, ensure, Result};

use crate::runtime::{InferenceSession, Tensor};

/// Frames per chunk (30 seconds at 50 fps).
const CHUNK_SIZE: usize = 1500;
/// Frames discarded from each edge of predictions.
const BORDER_SIZE: usize = 6;
/// Effective step between chunks.
const STRIDE: usize = CHUNK_SIZE - 2 * BORDER_SIZE;

/// Runs chunked beat/downbeat inference on a mel spectrogram.
///
/// The beat model was trained on 1500-frame segments. For longer audio,
/// the spectrogram is split into overlapping chunks, each run through
/// the model, and the results are aggregated using "keep_first" mode
/// (earlier chunks take priority in overlapping regions).
pub struct BeatInference<S: InferenceSession> {
    session: S,
}

impl<S: InferenceSession> BeatInference<S> {
    /// Wrap an already-loaded inference session for the beat model.
    pub fn new(session: S) -> Self {
        Self { session }
    }

    /// Get a mutable reference to the underlying session.
    pub fn session_mut(&mut self) -> &mut S {
        &mut self.session
    }

    /// Run inference on a full mel spectrogram.
    ///
    /// Input: mel spectrogram `Tensor` with shape `[1, T, 128]`.
    /// Returns: `(beat_logits, downbeat_logits)` — each `Vec<f32>` of length T.
    pub fn process(&mut self, mel: &Tensor) -> Result<(Vec<f32>, Vec<f32>)> {
        ensure!(
            mel.shape.len() == 3 && mel.shape[0] == 1 && mel.shape[2] == 128,
            "Expected mel shape [1, T, 128], got {:?}",
            mel.shape
        );

        let full_time = mel.shape[1];
        let starts = generate_starts(full_time);

        // Initialize with sentinel value; every frame will be overwritten.
        let mut beat_logits = vec![-1000.0f32; full_time];
        let mut downbeat_logits = vec![-1000.0f32; full_time];

        // Process in reverse order for "keep_first" aggregation:
        // earlier chunks are written last, so they overwrite later chunks
        // in overlapping regions.
        for &start in starts.iter().rev() {
            let chunk = extract_chunk(mel, start);

            let mut outputs = self
                .session
                .run(&[("spectrogram", &chunk)])?;

            let beat = extract_output(&mut outputs, "beat", "beat_logits")?;
            let downbeat = extract_output(&mut outputs, "downbeat", "downbeat_logits")?;

            // Strip border frames from both ends of the prediction.
            let valid_beat = &beat.data[BORDER_SIZE..CHUNK_SIZE - BORDER_SIZE];
            let valid_downbeat = &downbeat.data[BORDER_SIZE..CHUNK_SIZE - BORDER_SIZE];

            // Write valid predictions to output buffers.
            let write_start = (start + BORDER_SIZE as i32) as usize;
            for i in 0..STRIDE {
                let dest = write_start + i;
                if dest < full_time {
                    beat_logits[dest] = valid_beat[i];
                    downbeat_logits[dest] = valid_downbeat[i];
                }
            }
        }

        Ok((beat_logits, downbeat_logits))
    }
}

/// Extract a named output tensor, trying a fallback name if the primary is missing.
fn extract_output(
    outputs: &mut std::collections::HashMap<String, Tensor>,
    primary: &str,
    fallback: &str,
) -> Result<Tensor> {
    if let Some(t) = outputs.remove(primary) {
        return Ok(t);
    }
    if let Some(t) = outputs.remove(fallback) {
        return Ok(t);
    }
    Err(anyhow!(
        "Model missing output '{}' (also tried '{}'). Available: {:?}",
        primary,
        fallback,
        outputs.keys().collect::<Vec<_>>()
    ))
}

/// Generate chunk start positions for a spectrogram of `full_time` frames.
///
/// Starts at `-BORDER_SIZE`, steps by `STRIDE`, and adjusts the last start
/// to align with the spectrogram end (avoid_short_end).
fn generate_starts(full_time: usize) -> Vec<i32> {
    let mut starts = Vec::new();
    let mut pos = -(BORDER_SIZE as i32);
    let limit = full_time as i32 - BORDER_SIZE as i32;

    while pos < limit {
        starts.push(pos);
        pos += STRIDE as i32;
    }

    // Avoid short end: shift the last start so the final chunk aligns
    // with the spectrogram end.
    if full_time > STRIDE {
        if let Some(last) = starts.last_mut() {
            *last = full_time as i32 - (CHUNK_SIZE as i32 - BORDER_SIZE as i32);
        }
    }

    starts
}

/// Extract a single chunk from the mel spectrogram, zero-padding as needed.
///
/// Returns a `Tensor` of shape `[1, CHUNK_SIZE, 128]`.
fn extract_chunk(mel: &Tensor, start: i32) -> Tensor {
    let full_time = mel.shape[1];
    let n_mels = mel.shape[2]; // 128

    let actual_start = start.max(0) as usize;
    let actual_end = ((start + CHUNK_SIZE as i32) as usize).min(full_time);
    let pad_left = (-start).max(0) as usize;

    let mut data = vec![0.0f32; CHUNK_SIZE * n_mels];

    // Copy mel frames into the chunk at the correct offset.
    for t in actual_start..actual_end {
        let src_offset = t * n_mels;
        let dst_t = pad_left + (t - actual_start);
        let dst_offset = dst_t * n_mels;
        data[dst_offset..dst_offset + n_mels]
            .copy_from_slice(&mel.data[src_offset..src_offset + n_mels]);
    }

    Tensor {
        shape: vec![1, CHUNK_SIZE, n_mels],
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_starts_short() {
        // 100 frames — fits in a single chunk
        let starts = generate_starts(100);
        assert_eq!(starts, vec![-6]);
    }

    #[test]
    fn test_generate_starts_exact_chunk() {
        // Exactly 1500 frames — needs 2 chunks because after border trimming
        // the first chunk only covers frames 0..1488, leaving 12 frames uncovered.
        let starts = generate_starts(1500);
        assert_eq!(starts.len(), 2);
        assert_eq!(starts[0], -6);
        // Last adjusted: 1500 - (1500 - 6) = 6
        assert_eq!(starts[1], 6);
    }

    #[test]
    fn test_generate_starts_two_chunks() {
        // 2000 frames — needs two chunks
        let starts = generate_starts(2000);
        assert_eq!(starts.len(), 2);
        assert_eq!(starts[0], -6);
        // Last start adjusted: 2000 - (1500 - 6) = 506
        assert_eq!(starts[1], 506);
    }

    #[test]
    fn test_generate_starts_long() {
        // 5000 frames
        let starts = generate_starts(5000);
        assert_eq!(starts[0], -6);
        // Stride = 1488, so: -6, 1482, 2970, and then last adjusted to 5000-1494=3506
        assert_eq!(starts.len(), 4);
        assert_eq!(starts[1], 1482);
        assert_eq!(starts[2], 2970);
        assert_eq!(starts[3], 3506);
    }

    #[test]
    fn test_generate_starts_coverage() {
        // Verify every frame is covered by at least one chunk (after border trimming).
        for full_time in [50, 100, 500, 1488, 1500, 2000, 3000, 5000, 7800] {
            let starts = generate_starts(full_time);
            let mut covered = vec![false; full_time];
            for &start in &starts {
                let write_start = (start + BORDER_SIZE as i32).max(0) as usize;
                let write_end =
                    ((start + CHUNK_SIZE as i32 - BORDER_SIZE as i32) as usize).min(full_time);
                for i in write_start..write_end {
                    covered[i] = true;
                }
            }
            assert!(
                covered.iter().all(|&c| c),
                "Not all frames covered for full_time={full_time}. First uncovered: {}",
                covered.iter().position(|&c| !c).unwrap()
            );
        }
    }

    #[test]
    fn test_extract_chunk_first() {
        // First chunk (start = -6): should have 6 frames of left padding.
        let n_mels = 128;
        let full_time = 100;
        let mel = Tensor {
            shape: vec![1, full_time, n_mels],
            data: vec![1.0; full_time * n_mels],
        };

        let chunk = extract_chunk(&mel, -6);
        assert_eq!(chunk.shape, vec![1, CHUNK_SIZE, n_mels]);

        // First 6 frames should be zero (padding).
        for t in 0..6 {
            for f in 0..n_mels {
                assert_eq!(
                    chunk.data[t * n_mels + f],
                    0.0,
                    "Expected zero padding at t={t}, f={f}"
                );
            }
        }
        // Frame at t=6 should be 1.0 (from mel frame 0).
        assert_eq!(chunk.data[6 * n_mels], 1.0);
    }

    #[test]
    fn test_extract_chunk_middle() {
        // Middle chunk (start = 100, full_time = 5000): no padding.
        let n_mels = 128;
        let full_time = 5000;
        let mel = Tensor {
            shape: vec![1, full_time, n_mels],
            data: vec![1.0; full_time * n_mels],
        };

        let chunk = extract_chunk(&mel, 100);
        assert_eq!(chunk.shape, vec![1, CHUNK_SIZE, n_mels]);

        // All frames should be 1.0 (no padding needed).
        assert!(chunk.data.iter().all(|&v| v == 1.0));
    }
}
