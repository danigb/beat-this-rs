pub mod audio;
pub mod inference;
pub mod mel;
pub mod output;
pub mod postprocessing;
pub mod runtime;

use std::path::Path;

use anyhow::Result;

pub use audio::{load_audio, AudioData};
pub use inference::BeatInference;
pub use mel::MelProcessor;
pub use postprocessing::{BeatResult, PostProcessor};
pub use runtime::{InferenceRuntime, InferenceSession, Tensor};

/// Target sample rate expected by the mel spectrogram model.
const TARGET_SAMPLE_RATE: u32 = 22050;

/// High-level beat tracker composing the full pipeline.
///
/// Owns the mel spectrogram model, the beat inference model, and the
/// post-processor. Generic over the inference session type, so it works
/// with any backend (ort, rten, tract).
pub struct BeatThis<S: InferenceSession> {
    pub mel: MelProcessor<S>,
    pub inference: BeatInference<S>,
    pub post: PostProcessor,
}

impl<S: InferenceSession> BeatThis<S> {
    /// Create a new beat tracker by loading both ONNX models via the given runtime.
    ///
    /// - `runtime`: any `InferenceRuntime` (e.g. `OrtRuntime::default()`)
    /// - `mel_model_path`: path to the mel spectrogram ONNX model
    /// - `beat_model_path`: path to the beat tracking ONNX model
    pub fn new<R: InferenceRuntime<Session = S>>(
        runtime: &R,
        mel_model_path: &Path,
        beat_model_path: &Path,
    ) -> Result<Self> {
        let mel_session = runtime.load_model(mel_model_path)?;
        let beat_session = runtime.load_model(beat_model_path)?;

        Ok(Self {
            mel: MelProcessor::new(mel_session),
            inference: BeatInference::new(beat_session),
            post: PostProcessor::default(),
        })
    }

    /// Run the full pipeline on an audio file.
    ///
    /// Loads the file, resamples to 22050 Hz mono, computes mel spectrogram,
    /// runs beat inference, and post-processes into beat/downbeat timestamps.
    pub fn process_file(&mut self, path: &Path) -> Result<BeatResult> {
        let audio = load_audio(path, TARGET_SAMPLE_RATE)?;
        self.process_audio(&audio.samples, audio.sample_rate)
    }

    /// Run the full pipeline on raw audio samples.
    ///
    /// The samples are resampled to 22050 Hz if `sample_rate` differs.
    /// Input should be mono f32 PCM.
    pub fn process_audio(&mut self, samples: &[f32], sample_rate: u32) -> Result<BeatResult> {
        let samples = if sample_rate != TARGET_SAMPLE_RATE {
            audio::resample(samples.to_vec(), sample_rate, TARGET_SAMPLE_RATE)?
        } else {
            samples.to_vec()
        };

        let mel = self.mel.process(&samples)?;
        let (beat_logits, downbeat_logits) = self.inference.process(&mel)?;
        self.post.process(&beat_logits, &downbeat_logits)
    }
}
