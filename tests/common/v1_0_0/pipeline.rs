//! Verbatim transcription of the 1.0.0 pipeline (`src/lib.rs` lines 23-163 at 089b509).
//!
//! The only edits against the original are:
//! - type renames `BeatThis -> V1BeatThis`, `BeatAnalysis -> V1Analysis`,
//!   `TimedAnalysis -> V1TimedAnalysis`, `AnalysisTiming -> V1Timing`;
//! - the `use` lines below, which replace the crate-root ones (lines 1-21) so that the paths
//!   resolve against the modules mounted at the root of `tests/identity_v1_0_0.rs`.
//!
//! Function bodies are not touched. Never edit this file to make a test pass.

use std::path::Path;
use std::time::Duration;

use anyhow::Result;

use crate::{
    audio,
    audio::load_audio,
    inference::BeatPredictor,
    mel::MelExtractor,
    postprocessing::PeakPicker,
    runtime::{Model, Runtime, Tensor},
};

/// Target sample rate expected by the mel spectrogram model.
const TARGET_SAMPLE_RATE: u32 = 22050;

/// Full analysis result from the beat tracking pipeline.
#[derive(Debug, Clone)]
pub struct V1Analysis {
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
pub struct V1BeatThis<M: Model> {
    mel: MelExtractor<M>,
    predictor: BeatPredictor<M>,
    peak_picker: PeakPicker,
}

/// Per-stage timing from [`V1BeatThis::analyze_audio_timed`].
#[derive(Debug, Clone)]
pub struct V1Timing {
    pub mel: Duration,
    pub predict: Duration,
    pub decode: Duration,
}

/// Analysis result with optional per-stage timing.
#[derive(Debug, Clone)]
pub struct V1TimedAnalysis {
    pub analysis: V1Analysis,
    pub timing: V1Timing,
}

impl<M: Model> V1BeatThis<M> {
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
        let mel_model = runtime.load_model(mel_model_path)?;
        let beat_model = runtime.load_model(beat_model_path)?;

        Ok(Self {
            mel: MelExtractor::new(mel_model),
            predictor: BeatPredictor::new(beat_model),
            peak_picker: PeakPicker::default(),
        })
    }

    /// Create a beat tracker from pre-built models.
    ///
    /// Use this when you need separate runtimes for the mel and beat models
    /// (e.g. one with profiling enabled).
    pub fn from_models(mel_model: M, beat_model: M) -> Self {
        Self {
            mel: MelExtractor::new(mel_model),
            predictor: BeatPredictor::new(beat_model),
            peak_picker: PeakPicker::default(),
        }
    }

    /// Get a mutable reference to the beat prediction model.
    ///
    /// Useful for runtime-specific operations like ending ORT profiling.
    pub fn beat_model_mut(&mut self) -> &mut M {
        self.predictor.model_mut()
    }

    /// Run the full pipeline on raw audio samples.
    ///
    /// The samples are resampled to 22050 Hz if `sample_rate` differs.
    /// Input should be mono f32 PCM.
    pub fn analyze_audio(&mut self, samples: &[f32], sample_rate: u32) -> Result<V1Analysis> {
        Ok(self.analyze_audio_timed(samples, sample_rate)?.analysis)
    }

    /// Run the full pipeline on raw audio samples, returning per-stage timing.
    pub fn analyze_audio_timed(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<V1TimedAnalysis> {
        let samples = if sample_rate != TARGET_SAMPLE_RATE {
            audio::resample(samples.to_vec(), sample_rate, TARGET_SAMPLE_RATE)?
        } else {
            samples.to_vec()
        };

        let t = std::time::Instant::now();
        let mel = self.mel.extract(&samples)?;
        let mel_time = t.elapsed();

        let t = std::time::Instant::now();
        let (beat_logits, downbeat_logits) = self.predictor.predict(&mel)?;
        let predict_time = t.elapsed();

        let t = std::time::Instant::now();
        let (beats, downbeats) = self.peak_picker.decode(&beat_logits, &downbeat_logits)?;
        let decode_time = t.elapsed();

        Ok(V1TimedAnalysis {
            analysis: V1Analysis {
                beats,
                downbeats,
                mel,
                beat_logits,
                downbeat_logits,
            },
            timing: V1Timing {
                mel: mel_time,
                predict: predict_time,
                decode: decode_time,
            },
        })
    }

    /// Run the full pipeline on an audio file.
    ///
    /// Loads the file, resamples to 22050 Hz mono, computes mel spectrogram,
    /// runs beat prediction, and decodes into beat/downbeat timestamps.
    pub fn analyze_file(&mut self, path: &Path) -> Result<V1Analysis> {
        let audio = load_audio(path, TARGET_SAMPLE_RATE)?;
        self.analyze_audio(&audio.samples, audio.sample_rate)
    }
}
