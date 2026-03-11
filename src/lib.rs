mod audio;
mod inference;
mod mel;
mod output;
mod postprocessing;
mod runtime;

use std::path::Path;

use anyhow::Result;

pub use audio::{load_audio, AudioData};
pub use inference::BeatPredictor;
pub use mel::{num_frames as num_mel_frames, MelExtractor};
pub use output::*;
pub use postprocessing::PeakPicker;
pub use runtime::{ort::OrtRuntime, rten::RtenRuntime, Model, Runtime, Tensor};

/// Target sample rate expected by the mel spectrogram model.
const TARGET_SAMPLE_RATE: u32 = 22050;

/// Full analysis result from the beat tracking pipeline.
#[derive(Debug, Clone)]
pub struct BeatAnalysis {
    /// Beat times in seconds (sorted, deduplicated).
    pub beats: Vec<f32>,
    /// Downbeat times in seconds (sorted, deduplicated, snapped to nearest beat).
    pub downbeats: Vec<f32>,
    /// Mel spectrogram tensor with shape `[1, T, 128]` at 50 fps.
    pub mel: Tensor,
    /// Raw beat logits, one per spectrogram frame.
    pub beat_logits: Vec<f32>,
    /// Raw downbeat logits, one per spectrogram frame.
    pub downbeat_logits: Vec<f32>,
}

/// High-level beat tracker composing the full pipeline.
///
/// Owns the mel spectrogram model, the beat prediction model, and the
/// peak picker. Generic over the model type, so it works
/// with any backend (ort, rten, tract).
pub struct BeatThis<M: Model> {
    pub mel: MelExtractor<M>,
    pub predictor: BeatPredictor<M>,
    pub peak_picker: PeakPicker,
}

impl<M: Model> BeatThis<M> {
    /// Create a new beat tracker by loading both ONNX models via the given runtime.
    ///
    /// - `runtime`: any `Runtime` (e.g. `OrtRuntime::default()`)
    /// - `mel_model_path`: path to the mel spectrogram ONNX model
    /// - `beat_model_path`: path to the beat tracking ONNX model
    pub fn new<R: Runtime<Model = M>>(
        runtime: &R,
        mel_model_path: &Path,
        beat_model_path: &Path,
    ) -> Result<Self> {
        let mel_session = runtime.load_model(mel_model_path)?;
        let beat_session = runtime.load_model(beat_model_path)?;

        Ok(Self {
            mel: MelExtractor::new(mel_session),
            predictor: BeatPredictor::new(beat_session),
            peak_picker: PeakPicker::default(),
        })
    }

    /// Run the full pipeline on raw audio samples.
    ///
    /// The samples are resampled to 22050 Hz if `sample_rate` differs.
    /// Input should be mono f32 PCM.
    pub fn analyze_audio(&mut self, samples: &[f32], sample_rate: u32) -> Result<BeatAnalysis> {
        let samples = if sample_rate != TARGET_SAMPLE_RATE {
            audio::resample(samples.to_vec(), sample_rate, TARGET_SAMPLE_RATE)?
        } else {
            samples.to_vec()
        };

        let mel = self.mel.extract(&samples)?;
        let (beat_logits, downbeat_logits) = self.predictor.predict(&mel)?;
        let (beats, downbeats) = self.peak_picker.decode(&beat_logits, &downbeat_logits)?;

        Ok(BeatAnalysis {
            beats,
            downbeats,
            mel,
            beat_logits,
            downbeat_logits,
        })
    }

    /// Run the full pipeline on an audio file.
    ///
    /// Loads the file, resamples to 22050 Hz mono, computes mel spectrogram,
    /// runs beat prediction, and decodes into beat/downbeat timestamps.
    pub fn analyze_file(&mut self, path: &Path) -> Result<BeatAnalysis> {
        let audio = load_audio(path, TARGET_SAMPLE_RATE)?;
        self.analyze_audio(&audio.samples, audio.sample_rate)
    }
}
