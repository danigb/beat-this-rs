use std::f32::consts::PI;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{ensure, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use serde::Serialize;

use crate::BeatAnalysis;

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

/// Write a `.beats` file: tab-separated `time\tbeat_count` per line.
///
/// Beat count is 1 for downbeats, 2..N for subsequent beats in the measure.
/// Times are formatted with 3 decimal places.
pub fn write_beats_file(path: &Path, analysis: &BeatAnalysis) -> Result<()> {
    use std::io::Write;

    let counts = beat_counts(analysis);
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);

    for (&time, &count) in analysis.beats.iter().zip(counts.iter()) {
        writeln!(writer, "{:.3}\t{}", time, count)?;
    }

    Ok(())
}

/// Generate a click-track WAV file.
///
/// Downbeats get an 880 Hz click, regular beats get 440 Hz.
/// Each click is a 100ms sine wave with ADSR envelope.
/// Output is mono 44100 Hz 32-bit float WAV.
pub fn write_click_track(path: &Path, analysis: &BeatAnalysis) -> Result<()> {
    ensure!(
        !analysis.beats.is_empty(),
        "No beats to generate click track"
    );

    let counts = beat_counts(analysis);
    let total_duration = analysis.beats.last().unwrap() + CLICK_DURATION + CLICK_DECAY;
    let total_samples = (total_duration * CLICK_SAMPLE_RATE as f32) as usize;
    let mut buffer = vec![0.0f32; total_samples];

    for (&beat_time, &count) in analysis.beats.iter().zip(counts.iter()) {
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
    analysis: &BeatAnalysis,
    original_samples: &[f32],
    sample_rate: u32,
) -> Result<()> {
    ensure!(
        !analysis.beats.is_empty(),
        "No beats to generate mixed audio"
    );

    let counts = beat_counts(analysis);
    let original_duration = original_samples.len() as f32 / sample_rate as f32;
    let last_beat_end = analysis.beats.last().unwrap() + CLICK_DURATION + CLICK_DECAY;
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
    for (&beat_time, &count) in analysis.beats.iter().zip(counts.iter()) {
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

/// A single beat entry for JSON output.
#[derive(Serialize)]
pub struct BeatEntry {
    /// Beat timestamp in seconds.
    pub time: f32,
    /// Beat count within the measure (1 = downbeat).
    pub beat: i32,
    /// Whether this beat is a downbeat.
    pub downbeat: bool,
}

/// Top-level JSON output structure.
#[derive(Serialize)]
pub struct JsonOutput {
    pub beats: Vec<BeatEntry>,
    pub downbeats: Vec<f32>,
    pub bpm: Option<f32>,
}

/// Build the JSON output structure from a `BeatAnalysis`.
pub fn build_json_output(analysis: &BeatAnalysis) -> JsonOutput {
    let counts = beat_counts(analysis);
    let beats = analysis
        .beats
        .iter()
        .zip(counts.iter())
        .map(|(&time, &beat)| BeatEntry {
            time,
            beat,
            downbeat: beat == 1,
        })
        .collect();

    JsonOutput {
        beats,
        downbeats: analysis.downbeats.clone(),
        bpm: calculate_bpm(analysis),
    }
}

/// Write JSON output to stdout.
pub fn print_json_stdout(analysis: &BeatAnalysis) -> Result<()> {
    let output = build_json_output(analysis);
    let json = serde_json::to_string_pretty(&output)?;
    println!("{}", json);
    Ok(())
}

/// Write JSON output to a file.
pub fn write_json_file(path: &Path, analysis: &BeatAnalysis) -> Result<()> {
    let output = build_json_output(analysis);
    let file = std::fs::File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &output)?;
    Ok(())
}

/// Write the mel spectrogram as a numpy `.npy` file (v1.0 format).
///
/// The mel tensor has shape `[1, T, 128]`; the batch dimension is dropped and
/// the file contains a 2-D float32 array of shape `(T, 128)`, C-order.
/// The file can be loaded in Python with `numpy.load(path)`.
pub fn write_mel_npy(path: &Path, analysis: &BeatAnalysis) -> Result<()> {
    use std::io::Write;

    let mel = &analysis.mel;
    let t_frames = mel.shape[1];
    let n_mels = mel.shape[2];

    // Build the numpy dict header.
    let dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        t_frames, n_mels
    );

    // Pad with spaces so that total = 10 (prefix) + header_len is a multiple of 64.
    // header_len includes: dict + padding spaces + trailing '\n'.
    let base = 10 + dict.len() + 1; // 10-byte prefix + dict + '\n'
    let padding = (64 - base % 64) % 64;
    let header_len = (dict.len() + padding + 1) as u16;

    let file = std::fs::File::create(path)?;
    let mut f = BufWriter::new(file);

    f.write_all(b"\x93NUMPY")?; // magic
    f.write_all(&[1u8, 0u8])?; // version 1.0
    f.write_all(&header_len.to_le_bytes())?;
    f.write_all(dict.as_bytes())?;
    for _ in 0..padding {
        f.write_all(b" ")?;
    }
    f.write_all(b"\n")?;

    // Write [T, 128] float data (mel.data layout is [1, T, 128] row-major, batch=0).
    for &v in &mel.data[..t_frames * n_mels] {
        f.write_all(&v.to_le_bytes())?;
    }

    Ok(())
}

/// Per-file entry in batch summary JSON (no beat data, just metadata).
#[derive(Serialize)]
pub struct BatchFileEntry {
    pub input: String,
    pub duration_secs: f32,
    pub processing_time_secs: f32,
    pub outputs: Vec<String>,
}

/// Aggregate metrics for batch processing.
#[derive(Serialize)]
pub struct BatchSummary {
    pub total_files: usize,
    pub failed_files: usize,
    pub total_duration_secs: f32,
    pub total_processing_time_secs: f32,
    pub model_loading_time_secs: f32,
    pub realtime_factor: f32,
}

/// Top-level batch summary JSON (process metadata only).
#[derive(Serialize)]
pub struct BatchSummaryOutput {
    pub files: Vec<BatchFileEntry>,
    pub summary: BatchSummary,
}

/// Write batch summary as pretty-printed JSON to a file.
pub fn write_batch_json(path: &Path, output: &BatchSummaryOutput) -> Result<()> {
    let file = std::fs::File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, output)?;
    Ok(())
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
    use crate::runtime::Tensor;
    use std::io::Read;

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
    fn test_write_beats_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.beats");
        let analysis = make_analysis(vec![0.1, 0.6, 1.1], vec![0.1]);

        write_beats_file(&path, &analysis).unwrap();

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
        let analysis = make_analysis(vec![0.1, 0.6, 1.1], vec![0.1]);

        write_click_track(&path, &analysis).unwrap();

        // Verify WAV file is valid
        let reader = hound::WavReader::open(&path).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, CLICK_SAMPLE_RATE);
        assert_eq!(spec.sample_format, SampleFormat::Float);
        assert_eq!(spec.bits_per_sample, 32);

        // Verify non-zero samples exist
        let samples: Vec<f32> = reader.into_samples::<f32>().map(|s| s.unwrap()).collect();
        assert!(samples.iter().any(|&s| s.abs() > 0.01));
    }

    #[test]
    fn test_write_mixed_audio() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed.wav");
        let analysis = make_analysis(vec![0.1, 0.6], vec![0.1]);

        // 2 seconds of silence at 44100 Hz
        let original = vec![0.0f32; 44100 * 2];
        write_mixed_audio(&path, &analysis, &original, 44100).unwrap();

        let reader = hound::WavReader::open(&path).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, 44100);

        let samples: Vec<f32> = reader.into_samples::<f32>().map(|s| s.unwrap()).collect();
        // Should have at least as many samples as original
        assert!(samples.len() >= 44100 * 2);
        // Clicks should be audible (non-zero)
        assert!(samples.iter().any(|&s| s.abs() > 0.01));
    }

    #[test]
    fn test_calculate_bpm_120() {
        // Beats at 0.5s intervals = 120 BPM
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

    #[test]
    fn test_build_json_output() {
        let analysis = make_analysis(vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0], vec![0.5, 2.0]);
        let json_out = build_json_output(&analysis);

        assert_eq!(json_out.beats.len(), 6);
        assert_eq!(json_out.downbeats, vec![0.5, 2.0]);
        assert!(json_out.bpm.is_some());

        // First beat is downbeat
        assert_eq!(json_out.beats[0].beat, 1);
        assert!(json_out.beats[0].downbeat);

        // Second beat is not a downbeat
        assert_eq!(json_out.beats[1].beat, 2);
        assert!(!json_out.beats[1].downbeat);

        // Fourth beat is downbeat (2.0)
        assert_eq!(json_out.beats[3].beat, 1);
        assert!(json_out.beats[3].downbeat);

        // Serializes to valid JSON
        let json_str = serde_json::to_string(&json_out).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert!(parsed["beats"].is_array());
        assert!(parsed["downbeats"].is_array());
        assert!(parsed["bpm"].is_number());
    }

    #[test]
    fn test_write_mel_npy() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mel.npy");

        let t_frames = 10usize;
        let n_mels = 128usize;
        let data: Vec<f32> = (0..t_frames * n_mels).map(|i| i as f32).collect();
        let analysis = BeatAnalysis {
            beats: vec![],
            downbeats: vec![],
            mel: Tensor {
                shape: vec![1, t_frames, n_mels],
                data: data.clone(),
            },
            beat_logits: vec![],
            downbeat_logits: vec![],
        };

        write_mel_npy(&path, &analysis).unwrap();

        let bytes = std::fs::read(&path).unwrap();

        // Verify numpy magic
        assert_eq!(&bytes[0..6], b"\x93NUMPY");
        // Version 1.0
        assert_eq!(bytes[6], 1);
        assert_eq!(bytes[7], 0);

        // Total header size (10 bytes prefix) must be multiple of 64
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0);

        // Data starts at offset 10 + header_len
        let data_offset = 10 + header_len;
        assert_eq!(bytes.len(), data_offset + t_frames * n_mels * 4);

        // Verify a few float values
        let first = f32::from_le_bytes(bytes[data_offset..data_offset + 4].try_into().unwrap());
        assert_eq!(first, 0.0);
        let second =
            f32::from_le_bytes(bytes[data_offset + 4..data_offset + 8].try_into().unwrap());
        assert_eq!(second, 1.0);
    }

    #[test]
    fn test_write_batch_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("beat_this.json");

        let batch = BatchSummaryOutput {
            files: vec![
                BatchFileEntry {
                    input: "song1.mp3".to_string(),
                    duration_secs: 120.0,
                    processing_time_secs: 1.5,
                    outputs: vec!["song1.json".to_string(), "song1.beats".to_string()],
                },
                BatchFileEntry {
                    input: "song2.wav".to_string(),
                    duration_secs: 60.0,
                    processing_time_secs: 0.8,
                    outputs: vec!["song2.json".to_string()],
                },
            ],
            summary: BatchSummary {
                total_files: 2,
                failed_files: 0,
                total_duration_secs: 180.0,
                total_processing_time_secs: 2.3,
                model_loading_time_secs: 0.04,
                realtime_factor: 78.3,
            },
        };

        write_batch_json(&path, &batch).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

        assert_eq!(parsed["files"].as_array().unwrap().len(), 2);
        assert_eq!(parsed["files"][0]["input"], "song1.mp3");
        assert_eq!(parsed["files"][0]["duration_secs"], 120.0);
        assert_eq!(parsed["files"][0]["processing_time_secs"], 1.5);
        assert_eq!(
            parsed["files"][0]["outputs"],
            serde_json::json!(["song1.json", "song1.beats"])
        );
        assert!(parsed["files"][0]["beats"].is_null());
        assert!(parsed["files"][0]["bpm"].is_null());

        assert_eq!(parsed["summary"]["total_files"], 2);
        assert_eq!(parsed["summary"]["failed_files"], 0);
        assert_eq!(parsed["summary"]["total_duration_secs"], 180.0);
        assert_eq!(parsed["summary"]["realtime_factor"], 78.3);
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
        assert!(
            peak > 0.9,
            "Peak amplitude should be near 1.0, got {}",
            peak
        );
    }
}
