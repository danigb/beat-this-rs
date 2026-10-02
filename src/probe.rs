//! Unstable hooks for `examples/memory_probe.rs` and the identity tests.
//! Not part of the public API; may change in any release.
//!
//! The probe goes through this module (rather than copying `resample` and mimicking the copies in
//! the example) so that it keeps measuring the real code after later changes to it.

use anyhow::Result;

use crate::{BeatThis, Model, Tensor};

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

/// Build the mel input exactly as the pipeline does before running the graph (`mel.rs`
/// `extract_owned` and `runtime/rten.rs` `run`): the `Tensor` takes the samples by move and rten is
/// handed a borrowed view of them, so no PCM copy is made. Returns without running the graph; the
/// result is the number of bytes held.
///
/// Must be updated in lockstep whenever the real input path changes.
pub fn mel_input(samples: Vec<f32>) -> Result<usize> {
    let input = Tensor {
        shape: vec![1, samples.len()],
        data: samples,
    };
    let value = rten::ValueView::from_shape(input.shape.as_slice(), input.data.as_slice())
        .map_err(|e| anyhow::anyhow!("rten: failed to create input tensor: {e}"))?;
    std::hint::black_box(&value);
    Ok(std::hint::black_box(&input).data.len() * std::mem::size_of::<f32>())
}

/// The mel stage exactly as the pipeline runs it.
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
