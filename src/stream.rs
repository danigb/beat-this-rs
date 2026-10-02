//! Streaming front end: [`BeatStream`], created by [`BeatThis::stream`].

use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Result};

use crate::audio::{StreamResampler, RESAMPLE_CHUNK};
use crate::mel::{MelStream, MEL_STRIDE};
use crate::{BeatAnalysis, BeatThis, Model, TimedAnalysis, TARGET_SAMPLE_RATE};

/// Analysis of mono PCM fed in chunks of any size. Created by [`BeatThis::stream`].
///
/// Each [`push`](Self::push) resamples to 22 050 Hz in fixed 8192-frame chunks and runs the mel
/// spectrogram on every 64-frame-aligned window that is complete, keeping only the samples the next
/// window needs. [`finish`](Self::finish) flushes the resampler, computes the last window against
/// the true end of the signal, and runs the chunked beat model and peak picking on the whole mel.
///
/// **Output.** The result does not depend on how the signal is split into pushes, and it is bit for
/// bit the result of [`BeatThis::analyze_audio`] on the whole signal (which runs through this type).
/// Against 1.0.0 it is bit-identical for 22 050 Hz input and for 22 050 · 2^k Hz sources (11 025,
/// 44 100, 88 200 Hz). At other rates (48 kHz etc.) the resampled samples differ from 1.0.0's
/// one-shot resampler: slightly on short inputs, and more on long ones, where 1.0.0's resampler
/// loses position precision and this one does not; on 60 minutes at 48 kHz a few beats move by one
/// frame (see the CHANGELOG for 1.1.0).
///
/// **Memory.** Between pushes the stream holds O(chunk) of PCM state (about 3 MB, independent of the
/// input length) plus the mel frames computed so far, which grow by about 92 MB per hour of audio.
/// `finish` then adds the beat logits (two `f32` per mel frame) and the beat model's per-chunk work.
///
/// The stream borrows the [`BeatThis`] mutably for its lifetime, so the loaded models are reused;
/// it therefore cannot be stored in the same struct as that `BeatThis`. With the rten model it is
/// `Send`, so it can be moved to another thread together with that borrow. Dropping a stream
/// without calling `finish` is fine. After a `push` returns an error the stream is
/// unusable: every later `push` or `finish` returns an error.
pub struct BeatStream<'a, M: Model> {
    bt: &'a mut BeatThis<M>,
    /// `None` when the input is already at 22 050 Hz.
    resampler: Option<StreamResampler>,
    mel: MelStream,
    /// Resampler output for one chunk, handed to the mel stream and cleared; reused.
    scratch: Vec<f32>,
    /// Time spent in the mel stage over all pushes (resampling is not timed).
    mel_time: Duration,
    failed: bool,
}

impl<'a, M: Model> BeatStream<'a, M> {
    pub(crate) fn new(bt: &'a mut BeatThis<M>, sample_rate: u32) -> Result<Self> {
        ensure!(sample_rate > 0, "invalid sample rate: 0 Hz");
        let resampler = if sample_rate == TARGET_SAMPLE_RATE {
            None
        } else {
            Some(StreamResampler::new(sample_rate, TARGET_SAMPLE_RATE)?)
        };
        Ok(Self {
            bt,
            resampler,
            mel: MelStream::new(MEL_STRIDE),
            scratch: Vec::new(),
            mel_time: Duration::ZERO,
            failed: false,
        })
    }

    /// Feed the next mono `f32` samples, at the sample rate given to [`BeatThis::stream`].
    ///
    /// Any length works, including 0 and the whole signal at once: the input is processed in
    /// 8192-frame pieces, so a single large push does not hold extra copies of it.
    pub fn push(&mut self, mono: &[f32]) -> Result<()> {
        ensure!(!self.failed, "BeatStream used after an error");
        let result = self.push_inner(mono);
        self.failed = result.is_err();
        result
    }

    fn push_inner(&mut self, mono: &[f32]) -> Result<()> {
        match &mut self.resampler {
            Some(resampler) => {
                for piece in mono.chunks(RESAMPLE_CHUNK) {
                    resampler.push(piece, &mut self.scratch)?;
                    let t = Instant::now();
                    let pushed = self.mel.push(self.bt.mel.model_mut(), &self.scratch);
                    self.mel_time += t.elapsed();
                    self.scratch.clear();
                    pushed?;
                }
            }
            None => {
                let t = Instant::now();
                let pushed = self.mel.push(self.bt.mel.model_mut(), mono);
                self.mel_time += t.elapsed();
                pushed?;
            }
        }
        Ok(())
    }

    /// End the stream and return the analysis.
    ///
    /// Errors when nothing was pushed, or when the signal is too short to give any 22 050 Hz sample
    /// (as [`BeatThis::analyze_audio`] does for the same input).
    pub fn finish(self) -> Result<BeatAnalysis> {
        Ok(self.finish_timed()?.analysis)
    }

    /// As [`finish`](Self::finish), with per-stage timing. `timing.mel` is the mel time summed over
    /// every push and the end of the stream; resampling is not timed, as in
    /// [`BeatThis::analyze_audio_timed`].
    pub fn finish_timed(self) -> Result<TimedAnalysis> {
        let BeatStream {
            bt,
            resampler,
            mut mel,
            mut scratch,
            mut mel_time,
            failed,
        } = self;
        if failed {
            bail!("BeatStream used after an error");
        }
        if let Some(resampler) = resampler {
            resampler.finish(&mut scratch)?;
            let t = Instant::now();
            mel.push(bt.mel.model_mut(), &scratch)?;
            mel_time += t.elapsed();
        }
        drop(scratch);
        let t = Instant::now();
        let mel = mel.finish(bt.mel.model_mut())?;
        mel_time += t.elapsed();
        bt.predict_and_decode(mel, mel_time)
    }

    /// Floats of buffer capacity held between pushes outside the mel frames, and the capacity of
    /// the mel frames buffer. For the memory checks in `__probe`.
    pub(crate) fn retained_floats(&self) -> (usize, usize) {
        let resampler = self
            .resampler
            .as_ref()
            .map_or(0, StreamResampler::retained_floats);
        (
            resampler + self.mel.retained_floats() + self.scratch.capacity(),
            self.mel.frames_capacity(),
        )
    }
}
