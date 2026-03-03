use anyhow::{ensure, Result};

/// Default frames per second for the beat model.
const FPS: f32 = 50.0;

/// Beat detection results: sorted timestamps in seconds.
#[derive(Debug, Clone)]
pub struct BeatResult {
    /// Beat times in seconds (sorted, deduplicated).
    pub beats: Vec<f32>,
    /// Downbeat times in seconds (sorted, deduplicated, snapped to nearest beat).
    pub downbeats: Vec<f32>,
}

/// Post-processes raw beat/downbeat logits into timestamped events.
///
/// Applies max-pool peak picking, thresholding, deduplication, and
/// downbeat-to-beat alignment to convert per-frame logit vectors
/// into discrete beat and downbeat timestamps.
pub struct PostProcessor {
    fps: f32,
}

impl Default for PostProcessor {
    fn default() -> Self {
        Self { fps: FPS }
    }
}

impl PostProcessor {
    /// Create a new post-processor with the given frame rate.
    pub fn new(fps: f32) -> Self {
        Self { fps }
    }

    /// Process beat and downbeat logits into a `BeatResult`.
    ///
    /// Both input slices must have the same length (one value per spectrogram frame).
    /// Returns beat and downbeat times in seconds.
    pub fn process(&self, beat_logits: &[f32], downbeat_logits: &[f32]) -> Result<BeatResult> {
        ensure!(
            beat_logits.len() == downbeat_logits.len(),
            "beat_logits length ({}) != downbeat_logits length ({})",
            beat_logits.len(),
            downbeat_logits.len()
        );

        let beat_frames = find_peaks(beat_logits);
        let downbeat_frames = find_peaks(downbeat_logits);

        let beats: Vec<f32> = beat_frames.iter().map(|&f| f as f32 / self.fps).collect();
        let mut downbeats: Vec<f32> = downbeat_frames
            .iter()
            .map(|&f| f as f32 / self.fps)
            .collect();

        snap_downbeats_to_beats(&beats, &mut downbeats);

        Ok(BeatResult { beats, downbeats })
    }
}

/// Identify local maxima that exceed the logit threshold (> 0.0).
///
/// Uses a max-pool window of 7 frames (±3) with stride 1.
/// A frame is a peak if its value equals the local maximum and is positive.
fn find_peaks(logits: &[f32]) -> Vec<usize> {
    let len = logits.len();
    let mut peaks = Vec::new();

    for i in 0..len {
        // Threshold: logit > 0 corresponds to probability > 0.5 after sigmoid.
        if logits[i] <= 0.0 {
            continue;
        }

        // Max-pool: check if this frame is the maximum in a 7-frame window.
        let start = i.saturating_sub(3);
        let end = (i + 4).min(len);

        let mut is_max = true;
        for j in start..end {
            if logits[j] > logits[i] {
                is_max = false;
                break;
            }
        }

        if is_max {
            peaks.push(i);
        }
    }

    deduplicate_peaks(&peaks, 1)
}

/// Merge adjacent peak frame indices using a running mean.
///
/// Groups of consecutive peaks where each pair is at most `width` frames apart
/// are replaced by a single peak at their mean position (rounded).
fn deduplicate_peaks(peaks: &[usize], width: usize) -> Vec<usize> {
    if peaks.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut p = peaks[0] as f64;
    let mut c = 1.0_f64;

    for &p2_usize in &peaks[1..] {
        let p2 = p2_usize as f64;
        if p2 - p <= width as f64 {
            c += 1.0;
            p += (p2 - p) / c;
        } else {
            result.push(p.round() as usize);
            p = p2;
            c = 1.0;
        }
    }
    result.push(p.round() as usize);

    result
}

/// Snap each downbeat to the nearest beat time, then deduplicate.
fn snap_downbeats_to_beats(beat_times: &[f32], downbeat_times: &mut Vec<f32>) {
    if beat_times.is_empty() || downbeat_times.is_empty() {
        return;
    }

    for d_time in downbeat_times.iter_mut() {
        // Binary search for the closest beat.
        let pos = beat_times.partition_point(|&b| b < *d_time);

        let best = match (pos.checked_sub(1), beat_times.get(pos)) {
            (Some(before), Some(&after)) => {
                if (*d_time - beat_times[before]).abs() <= (after - *d_time).abs() {
                    beat_times[before]
                } else {
                    after
                }
            }
            (Some(before), None) => beat_times[before],
            (None, Some(&after)) => after,
            (None, None) => continue,
        };

        *d_time = best;
    }

    // Sort and deduplicate after snapping (multiple downbeats may map to the same beat).
    downbeat_times.sort_by(|a, b| a.total_cmp(b));
    downbeat_times.dedup();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_peaks_single_peak() {
        let logits = [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0];
        let peaks = find_peaks(&logits);
        assert_eq!(peaks, vec![3]);
    }

    #[test]
    fn test_find_peaks_below_threshold() {
        let logits = [-1.0, -0.5, -2.0, -0.1];
        let peaks = find_peaks(&logits);
        assert!(peaks.is_empty());
    }

    #[test]
    fn test_find_peaks_multiple_peaks() {
        // Two peaks separated by more than 3 frames.
        let mut logits = vec![0.0; 20];
        logits[3] = 2.0;
        logits[15] = 1.5;
        let peaks = find_peaks(&logits);
        assert_eq!(peaks, vec![3, 15]);
    }

    #[test]
    fn test_find_peaks_adjacent_equal() {
        // Adjacent frames with equal positive values — should deduplicate.
        let logits = [0.0, 1.0, 1.0, 0.0];
        let peaks = find_peaks(&logits);
        // Both pass max-pool check (tied), dedup merges them.
        assert_eq!(peaks.len(), 1);
        // Mean of indices 1 and 2 is 1.5, rounds to 2.
        assert_eq!(peaks[0], 2);
    }

    #[test]
    fn test_deduplicate_peaks_empty() {
        let peaks = deduplicate_peaks(&[], 1);
        assert!(peaks.is_empty());
    }

    #[test]
    fn test_deduplicate_peaks_no_adjacent() {
        let peaks = deduplicate_peaks(&[5, 10, 20], 1);
        assert_eq!(peaks, vec![5, 10, 20]);
    }

    #[test]
    fn test_deduplicate_peaks_merge() {
        // 10 and 11 merge (gap=1), mean=10.5. 12 is 1.5 from mean, starts new group.
        let peaks = deduplicate_peaks(&[10, 11, 12, 20], 1);
        assert_eq!(peaks, vec![11, 12, 20]);

        // Three truly adjacent peaks: 10, 11 merge to 10.5, then 11 is 0.5 from 10.5 → merges.
        let peaks = deduplicate_peaks(&[10, 11, 11, 20], 1);
        assert_eq!(peaks, vec![11, 20]);
    }

    #[test]
    fn test_deduplicate_peaks_single() {
        let peaks = deduplicate_peaks(&[42], 1);
        assert_eq!(peaks, vec![42]);
    }

    #[test]
    fn test_snap_downbeats() {
        let beats = vec![1.0, 2.0, 3.0];
        let mut downbeats = vec![1.1, 2.8];
        snap_downbeats_to_beats(&beats, &mut downbeats);
        assert_eq!(downbeats, vec![1.0, 3.0]);
    }

    #[test]
    fn test_snap_downbeats_dedup() {
        let beats = vec![1.0, 2.0, 3.0];
        // Both downbeats snap to 2.0.
        let mut downbeats = vec![1.8, 2.1];
        snap_downbeats_to_beats(&beats, &mut downbeats);
        assert_eq!(downbeats, vec![2.0]);
    }

    #[test]
    fn test_snap_downbeats_empty_beats() {
        let beats: Vec<f32> = vec![];
        let mut downbeats = vec![1.0, 2.0];
        snap_downbeats_to_beats(&beats, &mut downbeats);
        // Downbeats unchanged when no beats to snap to.
        assert_eq!(downbeats, vec![1.0, 2.0]);
    }

    #[test]
    fn test_snap_downbeats_empty_downbeats() {
        let beats = vec![1.0, 2.0];
        let mut downbeats: Vec<f32> = vec![];
        snap_downbeats_to_beats(&beats, &mut downbeats);
        assert!(downbeats.is_empty());
    }

    #[test]
    fn test_process_full() {
        // Construct logits with known peaks at specific frames.
        let mut beat_logits = vec![-5.0; 200];
        let mut downbeat_logits = vec![-5.0; 200];

        // Place beat peaks at frames 50, 100, 150.
        beat_logits[50] = 3.0;
        beat_logits[100] = 2.5;
        beat_logits[150] = 4.0;

        // Place downbeat peak at frame 51 (should snap to beat at frame 50).
        downbeat_logits[51] = 2.0;

        let pp = PostProcessor::new(50.0);
        let result = pp.process(&beat_logits, &downbeat_logits).unwrap();

        assert_eq!(result.beats, vec![1.0, 2.0, 3.0]); // 50/50, 100/50, 150/50
        assert_eq!(result.downbeats, vec![1.0]); // 51/50=1.02 snaps to 1.0
    }

    #[test]
    fn test_process_empty_logits() {
        let pp = PostProcessor::default();
        let result = pp.process(&[], &[]).unwrap();
        assert!(result.beats.is_empty());
        assert!(result.downbeats.is_empty());
    }

    #[test]
    fn test_process_mismatched_lengths() {
        let pp = PostProcessor::default();
        let err = pp.process(&[1.0, 2.0], &[1.0]);
        assert!(err.is_err());
    }
}
