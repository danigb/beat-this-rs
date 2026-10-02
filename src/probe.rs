//! Unstable hooks for `examples/memory_probe.rs` and the identity tests.
//! Not part of the public API; may change in any release.
//!
//! The probe goes through this module (rather than copying `resample` and mimicking the copies in
//! the example) so that it keeps measuring the real code after later changes to it.

use anyhow::Result;

use crate::{BeatAnalysis, BeatStream, BeatThis, Model, Tensor};

pub use crate::audio::StreamResampler;
pub use crate::mel::MelStream;

/// Deterministic test signal: a 220 Hz sine, an LCG noise bed at -20 dB and an 80 Hz decaying
/// pulse every 0.5 s. Yields the same samples whether pulled in one call or in chunks of any size.
pub struct Synth {
    rate: u32,
    i: u64,
    lcg: u32,
}

impl Synth {
    pub fn new(rate: u32) -> Self {
        Self {
            rate,
            i: 0,
            lcg: 0x1234_5678,
        }
    }

    /// Fill `out` with the next `out.len()` samples.
    pub fn fill(&mut self, out: &mut [f32]) {
        let rate = self.rate as f64;
        let pulse_period = (self.rate / 2) as u64;
        let two_pi = 2.0 * std::f64::consts::PI;
        for slot in out.iter_mut() {
            let t = self.i as f64 / rate;
            let tau = (self.i % pulse_period) as f64 / rate;
            self.lcg = self.lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = ((self.lcg >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0;
            let tone = 0.5 * (two_pi * 220.0 * t).sin();
            let pulse = 0.5 * (-tau / 0.05).exp() * (two_pi * 80.0 * tau).sin();
            *slot = (tone + pulse) as f32 + 0.05 * noise;
            self.i += 1;
        }
    }

    /// `n` samples from the start of the signal.
    pub fn take(rate: u32, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        Synth::new(rate).fill(&mut out);
        out
    }
}

/// Today's resample stage, exactly as the pipeline calls it.
pub fn resample(samples: Vec<f32>, source_sr: u32) -> Result<Vec<f32>> {
    crate::audio::resample(samples, source_sr, crate::TARGET_SAMPLE_RATE)
}

/// Build the mel input exactly as the whole-buffer mel stage does before its first graph run
/// (`mel.rs` `extract_owned` → `extract_windowed` → `run_window`, and `runtime/rten.rs` `run`;
/// since ticket 05 the pipeline itself goes through `BeatStream`, whose windows are the same): the mel graph
/// now runs per window, so the input is a `Tensor` holding a copy of the first window's samples
/// (`min(n, one window)`), and rten is handed a borrowed view of it. The whole signal stays alive
/// alongside, as it does in the pipeline. Returns without running the graph; the result is the
/// number of bytes held by the window tensor.
///
/// Must be updated in lockstep whenever the real input path changes.
pub fn mel_input(samples: Vec<f32>) -> Result<usize> {
    let len = crate::mel::first_window_len(samples.len());
    let input = Tensor {
        shape: vec![1, len],
        data: samples[..len].to_vec(),
    };
    let value = rten::ValueView::from_shape(input.shape.as_slice(), input.data.as_slice())
        .map_err(|e| anyhow::anyhow!("rten: failed to create input tensor: {e}"))?;
    std::hint::black_box(&value);
    std::hint::black_box(&samples);
    Ok(std::hint::black_box(&input).data.len() * std::mem::size_of::<f32>())
}

/// The whole-buffer mel stage: `MelExtractor::extract_owned`, which the pipeline ran before
/// ticket 05 routed `analyze_*` through `BeatStream`. Kept to measure and test the building block.
pub fn mel<M: Model>(bt: &mut BeatThis<M>, samples: Vec<f32>) -> Result<Tensor> {
    bt.mel.extract_owned(samples)
}

/// True when the chunked resampler is bit-identical to the one-shot call for this source rate
/// (to the pipeline's 22050 Hz target).
pub fn chunking_is_exact(source_sr: u32) -> bool {
    crate::audio::chunking_is_exact(source_sr, crate::TARGET_SAMPLE_RATE)
}

/// Output length of the one-shot resample call for `n` input frames.
pub fn one_shot_len(n: usize, source_sr: u32) -> usize {
    crate::audio::one_shot_len(n, crate::TARGET_SAMPLE_RATE as f64 / source_sr as f64)
}

/// Frames owned per mel window in the pipeline.
pub const MEL_STRIDE: usize = crate::mel::MEL_STRIDE;

/// Windowed mel extraction on a bare mel model with the given stride (a positive multiple of 64).
pub fn mel_windowed<M: Model>(model: &mut M, samples: &[f32], stride: usize) -> Result<Tensor> {
    crate::mel::extract_windowed(model, samples, stride)
}

/// The pipeline as `analyze_owned` ran it before ticket 05 routed it through `BeatStream`: the
/// whole signal resampled (`resample`), the windowed mel over the whole 22 050 Hz buffer
/// (`extract_owned`), then the beat model and peak picking. The identity tests compare `stream`
/// against it, which is what proves the routing bit-identical.
pub fn analyze_whole_buffer<M: Model>(
    bt: &mut BeatThis<M>,
    samples: Vec<f32>,
    sample_rate: u32,
) -> Result<BeatAnalysis> {
    let samples = if sample_rate != crate::TARGET_SAMPLE_RATE {
        crate::audio::resample(samples, sample_rate, crate::TARGET_SAMPLE_RATE)?
    } else {
        samples
    };
    let mel = bt.mel.extract_owned(samples)?;
    Ok(bt
        .predict_and_decode(mel, std::time::Duration::ZERO)?
        .analysis)
}

/// The beat model and peak picking on a mel spectrogram, as `BeatStream::finish` runs them.
pub fn predict_decode<M: Model>(bt: &mut BeatThis<M>, mel: Tensor) -> Result<BeatAnalysis> {
    Ok(bt
        .predict_and_decode(mel, std::time::Duration::ZERO)?
        .analysis)
}

/// The mel model of a `BeatThis`, for test-side compositions of the stream's building blocks.
pub fn mel_model<M: Model>(bt: &mut BeatThis<M>) -> &mut M {
    bt.mel.model_mut()
}

/// Floats of buffer capacity a `BeatStream` holds between pushes outside the mel frames (resampler
/// state, mel window buffer, scratch), and the capacity of its mel frames buffer.
pub fn stream_retained<M: Model>(stream: &BeatStream<'_, M>) -> (usize, usize) {
    stream.retained_floats()
}

/// Samples in the longest mel window of the pipeline; the mel stream's sample buffer is reserved
/// at this size and never grows.
pub const MEL_WINDOW_SAMPLES: usize = 441 * (crate::mel::MEL_STRIDE + 2 * 64 - 1);

/// Native frames the resampler takes per rubato call.
pub const RESAMPLE_CHUNK: usize = crate::audio::RESAMPLE_CHUNK;
