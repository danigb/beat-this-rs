use crate::BeatAnalysis;

/// Compute beat counts from beat and downbeat timestamps.
///
/// Faithful port of the Python reference `infer_beat_numbers`
/// (`refs/beat_this/beat_this/utils.py`): each downbeat is numbered `1`, beats within a
/// measure count upward, and a pickup measure (beats before the first downbeat) is
/// numbered so it leads *into* the first downbeat. Requires every downbeat to also be a
/// beat — guaranteed here because postprocessing snaps downbeats onto beat times.
///
/// Returns a `Vec<i32>` parallel to `analysis.beats`.
pub fn beat_counts(analysis: &BeatAnalysis) -> Vec<i32> {
    infer_beat_numbers(&analysis.beats, &analysis.downbeats)
}

/// Port of Python `infer_beat_numbers`. Assumes `beats` is sorted and `downbeats` is a
/// sorted subset of `beats` (the pipeline invariant).
fn infer_beat_numbers(beats: &[f32], downbeats: &[f32]) -> Vec<i32> {
    // Derive where to start counting, handling a pickup measure.
    let start_counter: i32 = if downbeats.len() >= 2 {
        // Index of each downbeat among the sorted beats (np.searchsorted / lower_bound).
        let first = beats.partition_point(|&b| b < downbeats[0]) as i32;
        let second = beats.partition_point(|&b| b < downbeats[1]) as i32;
        let beats_in_first_measure = second - first;
        let pickup_beats = first;
        if pickup_beats < beats_in_first_measure {
            beats_in_first_measure - pickup_beats
        } else {
            // More pickup beats than a full measure: don't try to estimate (Python warns).
            1
        }
    } else {
        // Fewer than two downbeats: can't estimate the pickup (Python warns).
        1
    };

    // Assemble the beat numbers. The increment/reset happens *before* the push, so a
    // non-downbeat first beat becomes `start_counter + 1` (matches Python exactly).
    let mut numbers = Vec::with_capacity(beats.len());
    let mut counter = start_counter;
    let mut db_idx = 0usize;
    for &beat in beats {
        match downbeats.get(db_idx) {
            Some(&db) if (beat - db).abs() < 0.001 => {
                counter = 1;
                db_idx += 1;
            }
            _ => counter += 1,
        }
        numbers.push(counter);
    }
    numbers
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
        // Fewer than two downbeats: Python can't estimate the pickup, so start_counter = 1
        // and the increment-before-append loop yields 2, 3, 4 (its warning says "start from 2").
        let analysis = make_analysis(vec![0.5, 1.0, 1.5], vec![]);
        let counts = beat_counts(&analysis);
        assert_eq!(counts, vec![2, 3, 4]);
    }

    #[test]
    fn test_beat_counts_beats_before_first_downbeat() {
        // Single downbeat with leading pickup beats: start_counter = 1, so the two pickup
        // beats are 2, 3, the downbeat resets to 1, and the next beat is 2.
        let analysis = make_analysis(vec![0.5, 1.0, 1.5, 2.0], vec![1.5]);
        let counts = beat_counts(&analysis);
        assert_eq!(counts, vec![2, 3, 1, 2]);
    }

    #[test]
    fn test_beat_counts_pickup_measure() {
        // 4/4 with two pickup beats: downbeats at index 2 and 6.
        let analysis = make_analysis(vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], vec![1.0, 3.0]);
        // beats_in_first_measure = 4, pickup_beats = 2 -> start_counter = 2 -> 3,4,1,2,...
        assert_eq!(beat_counts(&analysis), vec![3, 4, 1, 2, 3, 4, 1, 2]);
    }

    #[test]
    fn test_beat_counts_pickup_longer_than_first_measure() {
        // 3 pickup beats but only a 2-beat first measure -> start_counter falls back to 1.
        let analysis = make_analysis(vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], vec![1.5, 2.5]);
        // first=3, second=5 -> beats_in_first_measure=2, pickup_beats=3 (>=2) -> start_counter=1
        assert_eq!(beat_counts(&analysis), vec![2, 3, 4, 1, 2, 1, 2]);
    }

    #[test]
    fn test_beat_counts_clean_start_single_downbeat() {
        // Clean start (first beat is the downbeat): pickup_beats = 0, plain count up.
        let analysis = make_analysis(vec![0.5, 1.0, 1.5, 2.0], vec![0.5]);
        assert_eq!(beat_counts(&analysis), vec![1, 2, 3, 4]);
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
