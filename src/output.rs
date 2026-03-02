use std::f32::consts::PI;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{ensure, Result};
use hound::{SampleFormat, WavSpec, WavWriter};

use crate::postprocessing::BeatResult;

// Click synthesis constants
const CLICK_SAMPLE_RATE: u32 = 44100;
const CLICK_DURATION: f32 = 0.1; // 100ms
const CLICK_ATTACK: f32 = 0.01; // 10ms
const CLICK_DECAY: f32 = 0.05; // 50ms
const DOWNBEAT_FREQ: f32 = 880.0; // A5
const BEAT_FREQ: f32 = 440.0; // A4

// Mixing gains
const ORIGINAL_GAIN: f32 = 0.7;
const CLICK_GAIN: f32 = 0.3;

/// Compute beat counts from beat and downbeat timestamps.
///
/// Returns a `Vec<i32>` parallel to `result.beats`:
/// - 1 for downbeats
/// - 2, 3, 4, ... for subsequent beats within each measure
pub fn beat_counts(result: &BeatResult) -> Vec<i32> {
    let mut counts = Vec::with_capacity(result.beats.len());
    let mut counter = 0i32;

    for &beat_time in &result.beats {
        if is_downbeat(beat_time, &result.downbeats) {
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
    downbeats
        .iter()
        .any(|&db| (beat_time - db).abs() < 0.001)
}

/// Write a `.beats` file: tab-separated `time\tbeat_count` per line.
///
/// Beat count is 1 for downbeats, 2..N for subsequent beats in the measure.
/// Times are formatted with 3 decimal places.
pub fn write_beats_file(path: &Path, result: &BeatResult) -> Result<()> {
    use std::io::Write;

    let counts = beat_counts(result);
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);

    for (&time, &count) in result.beats.iter().zip(counts.iter()) {
        writeln!(writer, "{:.3}\t{}", time, count)?;
    }

    Ok(())
}

/// Generate a click-track WAV file.
///
/// Downbeats get an 880 Hz click, regular beats get 440 Hz.
/// Each click is a 100ms sine wave with ADSR envelope.
/// Output is mono 44100 Hz 32-bit float WAV.
pub fn write_click_track(path: &Path, result: &BeatResult) -> Result<()> {
    ensure!(!result.beats.is_empty(), "No beats to generate click track");

    let counts = beat_counts(result);
    let total_duration = result.beats.last().unwrap() + CLICK_DURATION + CLICK_DECAY;
    let total_samples = (total_duration * CLICK_SAMPLE_RATE as f32) as usize;
    let mut buffer = vec![0.0f32; total_samples];

    for (&beat_time, &count) in result.beats.iter().zip(counts.iter()) {
        let freq = if count == 1 { DOWNBEAT_FREQ } else { BEAT_FREQ };
        let click = generate_sine_click(freq, CLICK_SAMPLE_RATE);
        let start = (beat_time * CLICK_SAMPLE_RATE as f32) as usize;
        mix_into(&mut buffer, &click, start);
    }

    normalize(&mut buffer);
    write_wav(path, &buffer, CLICK_SAMPLE_RATE)
}

/// Generate a mixed WAV file: original audio + click track layered on top.
///
/// Original audio is scaled to 0.7 gain, clicks are added at 0.3 gain.
/// Output preserves the original sample rate. Output is mono.
pub fn write_mixed_audio(
    path: &Path,
    result: &BeatResult,
    original_samples: &[f32],
    sample_rate: u32,
) -> Result<()> {
    ensure!(!result.beats.is_empty(), "No beats to generate mixed audio");

    let counts = beat_counts(result);
    let original_duration = original_samples.len() as f32 / sample_rate as f32;
    let last_beat_end = result.beats.last().unwrap() + CLICK_DURATION + CLICK_DECAY;
    let total_duration = original_duration.max(last_beat_end);
    let total_samples = (total_duration * sample_rate as f32) as usize;

    let mut buffer = vec![0.0f32; total_samples];

    // Copy original audio scaled down
    for (i, &sample) in original_samples.iter().enumerate() {
        if i < buffer.len() {
            buffer[i] = sample * ORIGINAL_GAIN;
        }
    }

    // Add clicks
    for (&beat_time, &count) in result.beats.iter().zip(counts.iter()) {
        let freq = if count == 1 { DOWNBEAT_FREQ } else { BEAT_FREQ };
        let click = generate_sine_click(freq, sample_rate);
        let start = (beat_time * sample_rate as f32) as usize;
        mix_into_scaled(&mut buffer, &click, start, CLICK_GAIN);
    }

    normalize(&mut buffer);
    write_wav(path, &buffer, sample_rate)
}

/// Calculate BPM from beat timestamps using median inter-beat interval.
///
/// Filters out unrealistic intervals (<0.1s or >3.0s, i.e. outside 20–600 BPM).
/// Returns `None` if fewer than 2 valid intervals exist.
pub fn calculate_bpm(result: &BeatResult) -> Option<f32> {
    if result.beats.len() < 2 {
        return None;
    }

    let mut intervals: Vec<f32> = result
        .beats
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&iv| iv > 0.1 && iv < 3.0)
        .collect();

    if intervals.is_empty() {
        return None;
    }

    intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = intervals.len();
    let median = if n % 2 == 0 {
        (intervals[n / 2 - 1] + intervals[n / 2]) / 2.0
    } else {
        intervals[n / 2]
    };

    Some(60.0 / median)
}

/// Generate a single ADSR-enveloped sine click.
fn generate_sine_click(frequency: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (CLICK_DURATION * sample_rate as f32) as usize;
    let attack_samples = (CLICK_ATTACK * sample_rate as f32) as usize;
    let decay_samples = (CLICK_DECAY * sample_rate as f32) as usize;

    let mut waveform = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let amplitude = if i < attack_samples {
            // Attack: linear ramp up
            i as f32 / attack_samples as f32
        } else if i > num_samples - decay_samples {
            // Decay: linear ramp down
            (num_samples - i) as f32 / decay_samples as f32
        } else {
            1.0
        };
        waveform.push(amplitude * (2.0 * PI * frequency * t).sin());
    }

    waveform
}

/// Additively mix `src` into `dst` starting at `offset`.
fn mix_into(dst: &mut [f32], src: &[f32], offset: usize) {
    for (i, &sample) in src.iter().enumerate() {
        let pos = offset + i;
        if pos < dst.len() {
            dst[pos] += sample;
        }
    }
}

/// Additively mix `src` into `dst` starting at `offset`, scaled by `gain`.
fn mix_into_scaled(dst: &mut [f32], src: &[f32], offset: usize, gain: f32) {
    for (i, &sample) in src.iter().enumerate() {
        let pos = offset + i;
        if pos < dst.len() {
            dst[pos] += sample * gain;
        }
    }
}

/// Normalize buffer in-place if any sample exceeds ±1.0.
fn normalize(buffer: &mut [f32]) {
    let max_val = buffer.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    if max_val > 1.0 {
        let scale = 1.0 / max_val;
        for sample in buffer.iter_mut() {
            *sample *= scale;
        }
    }
}

/// Write a mono f32 WAV file.
fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let file = std::fs::File::create(path)?;
    let buf = BufWriter::new(file);
    let mut writer = WavWriter::new(buf, spec)?;

    for &sample in samples {
        writer.write_sample(sample)?;
    }

    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn make_result(beats: Vec<f32>, downbeats: Vec<f32>) -> BeatResult {
        BeatResult { beats, downbeats }
    }

    #[test]
    fn test_beat_counts_basic() {
        let result = make_result(vec![0.5, 1.0, 1.5, 2.0], vec![0.5]);
        let counts = beat_counts(&result);
        assert_eq!(counts, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_beat_counts_multiple_downbeats() {
        let result = make_result(
            vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            vec![0.5, 2.0],
        );
        let counts = beat_counts(&result);
        assert_eq!(counts, vec![1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn test_beat_counts_no_downbeats() {
        let result = make_result(vec![0.5, 1.0, 1.5], vec![]);
        let counts = beat_counts(&result);
        assert_eq!(counts, vec![1, 2, 3]);
    }

    #[test]
    fn test_beat_counts_beats_before_first_downbeat() {
        let result = make_result(vec![0.5, 1.0, 1.5, 2.0], vec![1.5]);
        let counts = beat_counts(&result);
        assert_eq!(counts, vec![1, 2, 1, 2]);
    }

    #[test]
    fn test_write_beats_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.beats");
        let result = make_result(vec![0.1, 0.6, 1.1], vec![0.1]);

        write_beats_file(&path, &result).unwrap();

        let mut content = String::new();
        std::fs::File::open(&path)
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();

        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "0.100\t1");
        assert_eq!(lines[1], "0.600\t2");
        assert_eq!(lines[2], "1.100\t3");
    }

    #[test]
    fn test_write_click_track() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clicks.wav");
        let result = make_result(vec![0.1, 0.6, 1.1], vec![0.1]);

        write_click_track(&path, &result).unwrap();

        // Verify WAV file is valid
        let reader = hound::WavReader::open(&path).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, CLICK_SAMPLE_RATE);
        assert_eq!(spec.sample_format, SampleFormat::Float);
        assert_eq!(spec.bits_per_sample, 32);

        // Verify non-zero samples exist
        let samples: Vec<f32> = reader
            .into_samples::<f32>()
            .map(|s| s.unwrap())
            .collect();
        assert!(samples.iter().any(|&s| s.abs() > 0.01));
    }

    #[test]
    fn test_write_mixed_audio() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed.wav");
        let result = make_result(vec![0.1, 0.6], vec![0.1]);

        // 2 seconds of silence at 44100 Hz
        let original = vec![0.0f32; 44100 * 2];
        write_mixed_audio(&path, &result, &original, 44100).unwrap();

        let reader = hound::WavReader::open(&path).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, 44100);

        let samples: Vec<f32> = reader
            .into_samples::<f32>()
            .map(|s| s.unwrap())
            .collect();
        // Should have at least as many samples as original
        assert!(samples.len() >= 44100 * 2);
        // Clicks should be audible (non-zero)
        assert!(samples.iter().any(|&s| s.abs() > 0.01));
    }

    #[test]
    fn test_calculate_bpm_120() {
        // Beats at 0.5s intervals = 120 BPM
        let result = make_result(
            vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            vec![0.0],
        );
        let bpm = calculate_bpm(&result).unwrap();
        assert!((bpm - 120.0).abs() < 0.1, "Expected ~120 BPM, got {}", bpm);
    }

    #[test]
    fn test_calculate_bpm_too_few_beats() {
        let result = make_result(vec![0.5], vec![]);
        assert!(calculate_bpm(&result).is_none());
    }

    #[test]
    fn test_calculate_bpm_empty() {
        let result = make_result(vec![], vec![]);
        assert!(calculate_bpm(&result).is_none());
    }

    #[test]
    fn test_generate_sine_click() {
        let click = generate_sine_click(440.0, 44100);
        let expected_len = (CLICK_DURATION * 44100.0) as usize;
        assert_eq!(click.len(), expected_len);

        // Starts near zero (attack)
        assert!(click[0].abs() < 0.01);
        // Ends near zero (decay)
        assert!(click.last().unwrap().abs() < 0.05);
        // Has significant amplitude in the middle
        let peak = click.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(peak > 0.9, "Peak amplitude should be near 1.0, got {}", peak);
    }
}
