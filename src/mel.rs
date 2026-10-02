use anyhow::{anyhow, bail, ensure, Result};

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

    /// Get a mutable reference to the underlying model.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Extract mel spectrogram from mono PCM samples at 22050 Hz.
    ///
    /// Input: mono f32 samples (any length); they are freed as soon as the mel is computed.
    /// Output: Tensor with shape `[1, time_frames, 128]`, with
    /// `time_frames = 1 + samples.len() / 441` (hop_length=441 for 50 fps at 22050 Hz).
    ///
    /// The graph runs in windows of `MEL_STRIDE` owned frames (see `extract_windowed`), so its
    /// intermediates are O(window) instead of O(signal). On rten the result is bit-identical to one
    /// whole-signal run of the graph (the 1.0.0 behaviour); other backends take the same windows,
    /// within their own numeric tolerance.
    pub fn extract_owned(&mut self, samples: Vec<f32>) -> Result<Tensor> {
        let mel = extract_windowed(&mut self.model, &samples, MEL_STRIDE)?;
        drop(samples);
        Ok(mel)
    }
}

/// Samples in the first graph run of [`MelExtractor::extract_owned`] for an `n`-sample signal.
pub(crate) fn first_window_len(n: usize) -> usize {
    n.min(interior_end_sample(0, MEL_STRIDE))
}

/// Mel graph frame step (50 fps at 22 050 Hz).
const HOP: usize = 441;
/// Mel bins per frame.
const N_MELS: usize = 128;
/// Every rten-gemm 0.24 f32 kernel's `NR` (4, 8, 16, 32) divides this. Windows start at a multiple
/// of it and interior windows span a multiple of it, so each owned frame lands in a full column
/// tile, or in the partial tile exactly when it does in the whole-signal run. A window of any other
/// alignment can differ from the whole-signal run by 1 ULP in a few values (rten fuses the
/// `Mul` scales into the mel `MatMul`, and its partial tiles round differently).
const MEL_ALIGN: usize = 64;
/// Frames of context on each side of a window: at least 2 (the first and last frames of a graph
/// run see its reflect padding) and equal to `MEL_ALIGN`, which keeps window starts aligned.
const MEL_HALO: usize = MEL_ALIGN;
/// Frames owned per window; a multiple of `MEL_ALIGN`. 1536 frames is about 30.7 s.
pub(crate) const MEL_STRIDE: usize = 1536;

/// One graph run of the window plan. Window `k` owns frames `[k*stride, (k+1)*stride)` of a signal
/// with `T = 1 + n/HOP` frames.
struct Window {
    /// First sample of the window.
    start: usize,
    /// One past the last sample of the window.
    end: usize,
    /// Frames the graph returns for `[start, end)`.
    frames: usize,
    /// Owned frames, local to the window: `[keep.0, keep.1)`.
    keep: (usize, usize),
}

/// A positive multiple of `MEL_ALIGN` (a power of two), tested with a mask.
fn valid_stride(stride: usize) -> bool {
    stride > 0 && stride & (MEL_ALIGN - 1) == 0
}

/// First frame of window `k`; always a multiple of `MEL_ALIGN`.
fn window_start_frame(k: usize, stride: usize) -> usize {
    (k * stride).saturating_sub(MEL_HALO)
}

/// One past the last sample an interior window `k` reads.
fn interior_end_sample(k: usize, stride: usize) -> usize {
    HOP * ((k + 1) * stride + MEL_HALO - 1)
}

/// Window `k` when `interior_end_sample(k) <= n`: it returns `(k+1)*stride + HALO - o` frames (a
/// multiple of `MEL_ALIGN`) and owns its middle `stride` of them.
fn interior_window(k: usize, stride: usize) -> Window {
    let o = window_start_frame(k, stride);
    Window {
        start: HOP * o,
        end: interior_end_sample(k, stride),
        frames: (k + 1) * stride + MEL_HALO - o,
        keep: (k * stride - o, (k + 1) * stride - o),
    }
}

/// Window `k` when it is the last one: it runs to the end of the `n`-sample signal, so it sees the
/// same end padding as the whole-signal run. For `k = 0` it is the whole signal.
fn last_window(k: usize, stride: usize, n: usize) -> Window {
    let o = window_start_frame(k, stride);
    let total = 1 + n / HOP;
    Window {
        start: HOP * o,
        end: n,
        frames: total - o,
        keep: (k * stride - o, total - o),
    }
}

/// Run the mel graph on `slice` and check that it returns `expect_frames` frames.
fn run_window<M: Model>(model: &mut M, slice: &[f32], expect_frames: usize) -> Result<Tensor> {
    let input = Tensor {
        shape: vec![1, slice.len()],
        data: slice.to_vec(),
    };
    let mut outputs = model.run(&[("audio_pcm", &input)])?;
    let mel = outputs
        .remove("mel_spectrogram")
        .ok_or_else(|| anyhow!("Model missing 'mel_spectrogram' output"))?;
    ensure!(
        mel.shape.len() == 3 && mel.shape[0] == 1 && mel.shape[2] == N_MELS,
        "Unexpected mel shape: {:?}",
        mel.shape
    );
    ensure!(
        mel.shape[1] == expect_frames,
        "mel graph returned {} frames for {} samples, expected {} (is the mel model's hop {}?)",
        mel.shape[1],
        slice.len(),
        expect_frames,
        HOP
    );
    Ok(mel)
}

/// Append the owned frames of `mel` (the output of `window`) to `frames`.
fn keep_frames(frames: &mut Vec<f32>, mel: &Tensor, window: &Window) {
    frames.extend_from_slice(&mel.data[window.keep.0 * N_MELS..window.keep.1 * N_MELS]);
}

/// Mel spectrogram of an in-memory signal, computed in 64-aligned windows of `stride` owned frames.
///
/// Bit-identical to one whole-signal run of the graph on rten (see `MEL_ALIGN`); the other
/// backends use the same plan, within their numeric tolerance. `stride` must be a positive multiple
/// of `MEL_ALIGN`.
pub(crate) fn extract_windowed<M: Model>(
    model: &mut M,
    samples: &[f32],
    stride: usize,
) -> Result<Tensor> {
    assert!(valid_stride(stride), "bad mel stride");
    let n = samples.len();
    let total = 1 + n / HOP;
    let mut frames = Vec::with_capacity(total * N_MELS);
    let mut k = 0;
    loop {
        let interior = interior_end_sample(k, stride) <= n;
        let window = if interior {
            interior_window(k, stride)
        } else {
            last_window(k, stride, n)
        };
        let mel = run_window(model, &samples[window.start..window.end], window.frames)?;
        keep_frames(&mut frames, &mel, &window);
        if !interior {
            break;
        }
        k += 1;
    }
    Ok(Tensor {
        shape: vec![1, total, N_MELS],
        data: frames,
    })
}

/// Push-based mel spectrogram: keeps only the samples the next window needs, so memory is
/// O(window) plus the mel frames emitted so far. Output is identical to `extract_windowed` with
/// the same stride, whatever the push sizes.
pub struct MelStream {
    stride: usize,
    /// Next window index.
    k: usize,
    /// Samples from global index `buf_start`.
    buf: Vec<f32>,
    /// Always `HOP * window_start_frame(k)`.
    buf_start: usize,
    /// Samples pushed so far.
    total: usize,
    /// Emitted frames, row-major `[t][128]`.
    frames: Vec<f32>,
}

impl MelStream {
    /// `stride` must be a positive multiple of `MEL_ALIGN`.
    pub fn new(stride: usize) -> Self {
        assert!(valid_stride(stride), "bad mel stride");
        Self {
            stride,
            k: 0,
            // The longest window (any window after the first), reserved once so the buffer never
            // grows by doubling.
            buf: Vec::with_capacity(HOP * (stride + 2 * MEL_HALO - 1)),
            buf_start: 0,
            total: 0,
            frames: Vec::new(),
        }
    }

    /// Frames emitted so far.
    pub fn frames_emitted(&self) -> usize {
        self.frames.len() / N_MELS
    }

    /// Floats of sample-buffer capacity held between calls: one window's samples, reserved up
    /// front, whatever the input length. The emitted mel frames are not counted (see [`Self::frames_capacity`]).
    pub(crate) fn retained_floats(&self) -> usize {
        self.buf.capacity()
    }

    /// Floats of capacity of the emitted-frames buffer.
    pub(crate) fn frames_capacity(&self) -> usize {
        self.frames.capacity()
    }

    /// Run window `self.k` over the buffer, keep its owned frames, and drop the samples that the
    /// next window does not need.
    fn run_buffered<M: Model>(&mut self, model: &mut M, window: &Window) -> Result<()> {
        let slice = &self.buf[window.start - self.buf_start..window.end - self.buf_start];
        let mel = run_window(model, slice, window.frames)?;
        keep_frames(&mut self.frames, &mel, window);
        self.k += 1;
        let next_start = HOP * window_start_frame(self.k, self.stride);
        self.buf.drain(..next_start - self.buf_start);
        self.buf_start = next_start;
        Ok(())
    }

    /// Feed samples; runs every window that is complete.
    pub fn push<M: Model>(&mut self, model: &mut M, samples: &[f32]) -> Result<()> {
        self.total += samples.len();
        let mut rest = samples;
        loop {
            let end = interior_end_sample(self.k, self.stride);
            let have = self.buf_start + self.buf.len();
            if have < end {
                let take = (end - have).min(rest.len());
                self.buf.extend_from_slice(&rest[..take]);
                rest = &rest[take..];
                if self.buf_start + self.buf.len() < end {
                    return Ok(());
                }
            }
            let window = interior_window(self.k, self.stride);
            self.run_buffered(model, &window)?;
            debug_assert!(self.buf.len() <= HOP * (self.stride + 2 * MEL_HALO));
        }
    }

    /// Apply the end-of-signal rules and return the whole mel `[1, T, 128]`. Errors when nothing
    /// was pushed, as the whole-signal run does.
    pub fn finish<M: Model>(mut self, model: &mut M) -> Result<Tensor> {
        let n = self.total;
        if n == 0 {
            bail!("cannot compute a mel spectrogram of empty input");
        }
        // Interior windows are run by `push` as soon as they are complete; this only catches a
        // window that became interior without a later push.
        while interior_end_sample(self.k, self.stride) <= n {
            let window = interior_window(self.k, self.stride);
            self.run_buffered(model, &window)?;
        }
        let window = last_window(self.k, self.stride, n);
        let slice = &self.buf[window.start - self.buf_start..];
        let mel = run_window(model, slice, window.frames)?;
        keep_frames(&mut self.frames, &mel, &window);
        let total = 1 + n / HOP;
        ensure!(
            self.frames.len() == total * N_MELS,
            "mel stream emitted {} frames, expected {}",
            self.frames.len() / N_MELS,
            total
        );
        Ok(Tensor {
            shape: vec![1, total, N_MELS],
            data: self.frames,
        })
    }
}
