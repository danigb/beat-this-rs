use anyhow::{anyhow, ensure, Result};

use crate::runtime::{Model, Tensor};

/// Computes log-mel spectrograms via an ONNX model.
///
/// The model takes raw PCM audio and returns a mel spectrogram,
/// guaranteeing exact numerical parity with the Python training pipeline.
pub struct MelExtractor<M: Model> {
    model: M,
}

impl<M: Model> MelExtractor<M> {
    /// Wrap an already-loaded model for mel spectrogram extraction.
    pub fn new(model: M) -> Self {
        Self { model }
    }

    /// Extract mel spectrogram from mono PCM samples at 22050 Hz.
    ///
    /// Input: mono f32 samples (any length).
    /// Output: Tensor with shape `[1, time_frames, 128]`.
    ///
    /// The number of time frames depends on sample count:
    /// `time_frames ≈ samples.len() / 441` (hop_length=441 for 50 fps at 22050 Hz).
    pub fn extract(&mut self, samples: &[f32]) -> Result<Tensor> {
        let input = Tensor {
            shape: vec![1, samples.len()],
            data: samples.to_vec(),
        };

        let mut outputs = self.model.run(&[("audio_pcm", &input)])?;

        let mel = outputs
            .remove("mel_spectrogram")
            .ok_or_else(|| anyhow!("Model missing 'mel_spectrogram' output"))?;

        ensure!(
            mel.shape.len() == 3 && mel.shape[0] == 1 && mel.shape[2] == 128,
            "Unexpected mel shape: {:?}",
            mel.shape
        );

        Ok(mel)
    }
}
