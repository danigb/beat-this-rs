use crate::BeatAnalysis;

/// Compute beat counts from beat and downbeat timestamps.
///
/// Returns a `Vec<i32>` parallel to `analysis.beats`:
/// - 1 for downbeats
/// - 2, 3, 4, ... for subsequent beats within each measure
pub fn beat_counts(analysis: &BeatAnalysis) -> Vec<i32> {
    let mut counts = Vec::with_capacity(analysis.beats.len());
    let mut counter = 0i32;

    for &beat_time in &analysis.beats {
        if is_downbeat(beat_time, &analysis.downbeats) {
            counter = 1;
        } else {
            counter += 1;
        }
        counts.push(counter);
    }

    counts
}

/// Check if a beat time corresponds to a downbeat (within tolerance).
fn is_downbeat(beat_time: f32, downbeats: &[f32]) -> bool {
    downbeats.iter().any(|&db| (beat_time - db).abs() < 0.001)
}

/// Calculate BPM from beat timestamps using median inter-beat interval.
///
/// Filters out unrealistic intervals (<0.1s or >3.0s, i.e. outside 20–600 BPM).
/// Returns `None` if fewer than 2 valid intervals exist.
pub fn calculate_bpm(analysis: &BeatAnalysis) -> Option<f32> {
    if analysis.beats.len() < 2 {
        return None;
    }

    let mut intervals: Vec<f32> = analysis
        .beats
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&iv| iv > 0.1 && iv < 3.0)
        .collect();

    if intervals.is_empty() {
        return None;
    }

    intervals.sort_by(|a, b| a.total_cmp(b));
    let n = intervals.len();
    let median = if n.is_multiple_of(2) {
        (intervals[n / 2 - 1] + intervals[n / 2]) / 2.0
    } else {
        intervals[n / 2]
    };

    Some(60.0 / median)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Tensor;

    fn make_analysis(beats: Vec<f32>, downbeats: Vec<f32>) -> BeatAnalysis {
        BeatAnalysis {
            beats,
            downbeats,
            mel: Tensor {
                shape: vec![1, 0, 128],
                data: vec![],
            },
            beat_logits: vec![],
            downbeat_logits: vec![],
        }
    }

    #[test]
    fn test_beat_counts_basic() {
        let analysis = make_analysis(vec![0.5, 1.0, 1.5, 2.0], vec![0.5]);
        let counts = beat_counts(&analysis);
        assert_eq!(counts, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_beat_counts_multiple_downbeats() {
        let analysis = make_analysis(vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0], vec![0.5, 2.0]);
        let counts = beat_counts(&analysis);
        assert_eq!(counts, vec![1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn test_beat_counts_no_downbeats() {
        let analysis = make_analysis(vec![0.5, 1.0, 1.5], vec![]);
        let counts = beat_counts(&analysis);
        assert_eq!(counts, vec![1, 2, 3]);
    }

    #[test]
    fn test_beat_counts_beats_before_first_downbeat() {
        let analysis = make_analysis(vec![0.5, 1.0, 1.5, 2.0], vec![1.5]);
        let counts = beat_counts(&analysis);
        assert_eq!(counts, vec![1, 2, 1, 2]);
    }

    #[test]
    fn test_calculate_bpm_120() {
        let analysis = make_analysis(vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], vec![0.0]);
        let bpm = calculate_bpm(&analysis).unwrap();
        assert!((bpm - 120.0).abs() < 0.1, "Expected ~120 BPM, got {}", bpm);
    }

    #[test]
    fn test_calculate_bpm_too_few_beats() {
        let analysis = make_analysis(vec![0.5], vec![]);
        assert!(calculate_bpm(&analysis).is_none());
    }

    #[test]
    fn test_calculate_bpm_empty() {
        let analysis = make_analysis(vec![], vec![]);
        assert!(calculate_bpm(&analysis).is_none());
    }
}
