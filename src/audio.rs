use anyhow::{anyhow, bail, Result};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use std::fs::File;
use std::path::Path;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// Mono audio data at a known sample rate.
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Load an audio file, convert to mono, and resample to `target_sr`.
///
/// Supports MP3, WAV, FLAC, OGG, and other formats via symphonia.
/// Uses high-quality sinc resampling via rubato when the source rate
/// differs from `target_sr`.
pub fn load_audio(path: &Path, target_sr: u32) -> Result<AudioData> {
    let (samples, source_sr, channels) = decode(path)?;
    let mono = to_mono(&samples, channels);
    let resampled = resample(mono, source_sr, target_sr)?;

    Ok(AudioData {
        samples: resampled,
        sample_rate: target_sr,
    })
}

/// Decode an audio file into interleaved f32 samples.
/// Returns (samples, sample_rate, channel_count).
fn decode(path: &Path) -> Result<(Vec<f32>, u32, usize)> {
    let src = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    // Provide file extension hint for better format detection
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe().probe(
        &hint,
        mss,
        FormatOptions::default(),
        MetadataOptions::default(),
    )?;

    let track = format
        .first_track_known_codec(TrackType::Audio)
        .ok_or_else(|| anyhow!("no supported audio tracks"))?;
    let track_id = track.id;
    let codec_params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| anyhow!("audio track missing codec parameters"))?;

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(codec_params, &AudioDecoderOptions::default())?;

    let mut samples: Vec<f32> = Vec::new();
    let mut scratch: Vec<f32> = Vec::new();
    let mut source_sr = 0u32;
    let mut channels = 0usize;

    // `next_packet` returns `Ok(None)` at end of stream in symphonia 0.6.
    while let Some(packet) = format.next_packet()? {
        if packet.track_id != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = decoded.spec();
                source_sr = spec.rate();
                channels = spec.channels().count();

                // `copy_to_vec_interleaved` resizes `scratch` to exactly this
                // packet's sample count, so we append it to the running buffer.
                decoded.copy_to_vec_interleaved(&mut scratch);
                samples.extend_from_slice(&scratch);
            }
            Err(Error::DecodeError(_)) => (), // skip corrupted packets
            Err(err) => return Err(anyhow!(err)),
        }
    }

    if source_sr == 0 {
        return Err(anyhow!("failed to decode any audio packets"));
    }

    Ok((samples, source_sr, channels))
}

/// Convert interleaved multi-channel audio to mono by averaging channels.
fn to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }
    samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// The sinc resampler parameters, shared by [`resample`] and [`StreamResampler`] so that they
/// cannot diverge.
fn sinc_params() -> SincInterpolationParameters {
    SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    }
}

/// Input chunk fed to rubato by [`StreamResampler`]. Keeps rubato's buffers to ~70 KB.
const RESAMPLE_CHUNK: usize = 8192;

/// Output length of the one-shot [`resample`] call for `n` input frames: rubato 3.0.0
/// `Async::calculate_output_size` (asynchro.rs:384-386) with `chunk_size = n`,
/// `interpolator_len = 256` and `last_index = -(256 - 1)`. Negative values saturate to 0, as there.
pub(crate) fn one_shot_len(n: usize, ratio: f64) -> usize {
    ((n as f64 - (256 + 1) as f64 - (-(256.0 - 1.0))) * (0.5 * ratio + 0.5 * ratio)).floor()
        as usize
}

/// True when chunked processing is provably bit-identical to the one-shot call: rubato's
/// per-output step `t = 1/ratio` (computed exactly as rubato does) is a dyadic rational with at
/// most 20 fractional bits, so every read position `-255 + k*t` is exact in f64 and no addition
/// rounds, whatever the chunking. True for sources 22050 * 2^k (11025, 44100, 88200, 176400).
pub(crate) fn chunking_is_exact(source_sr: u32, target_sr: u32) -> bool {
    let ratio = target_sr as f64 / source_sr as f64;
    let t = 1.0 / ratio;
    (t * 1_048_576.0).fract() == 0.0 && t < 1024.0
}

/// Push-based sinc resampler with the one-shot [`resample`]'s parameters, leading delay and
/// output length.
///
/// Feeds rubato fixed chunks of [`RESAMPLE_CHUNK`] frames, so memory is O(chunk) instead of
/// O(signal). Output is bit-identical to the one-shot call when [`chunking_is_exact`], and drifts
/// slightly otherwise (rubato advances its read position as an `f64` and renormalises it per chunk).
/// The output does not depend on how the input is split across `push` calls.
pub struct StreamResampler {
    inner: Async<f32>,
    ratio: f64,
    /// Fewer than `RESAMPLE_CHUNK` native samples waiting for a full chunk.
    pending: Vec<f32>,
    /// Computed, not yet released.
    produced: Vec<f32>,
    out_buf: Vec<f32>,
    total_in: usize,
    released: usize,
}

impl StreamResampler {
    pub fn new(source_sr: u32, target_sr: u32) -> Result<Self> {
        let ratio = target_sr as f64 / source_sr as f64;
        let inner = Async::<f32>::new_sinc(
            ratio,
            2.0,
            &sinc_params(),
            RESAMPLE_CHUNK,
            1, // mono
            FixedAsync::Input,
        )?;
        let out_buf = vec![0.0; inner.output_frames_max()];
        Ok(Self {
            inner,
            ratio,
            pending: Vec::with_capacity(RESAMPLE_CHUNK),
            produced: Vec::new(),
            out_buf,
            total_in: 0,
            released: 0,
        })
    }

    /// Run one rubato call on `chunk` (exactly `RESAMPLE_CHUNK` frames, of which the first
    /// `partial` are valid when given) and append its output to `produced`.
    fn run(&mut self, chunk: &[f32], partial: Option<usize>) -> Result<()> {
        let input = InterleavedSlice::new(chunk, 1, RESAMPLE_CHUNK)
            .map_err(|e| anyhow!("resampler input adapter: {e:?}"))?;
        let frames_out = self.inner.output_frames_next();
        let mut output = InterleavedSlice::new_mut(&mut self.out_buf, 1, frames_out)
            .map_err(|e| anyhow!("resampler output adapter: {e:?}"))?;
        let indexing = partial.map(|n| Indexing {
            input_offset: 0,
            output_offset: 0,
            partial_len: Some(n),
            active_channels_mask: None,
        });
        let (_, written) =
            self.inner
                .process_into_buffer(&input, &mut output, indexing.as_ref())?;
        self.produced.extend_from_slice(&self.out_buf[..written]);
        Ok(())
    }

    /// Move what is releasable from `produced` to `out`: never more than the one-shot length for
    /// the input seen so far, which the final length can only exceed.
    fn release(&mut self, out: &mut Vec<f32>) {
        let cap = one_shot_len(self.total_in, self.ratio);
        let k = cap.saturating_sub(self.released).min(self.produced.len());
        out.extend(self.produced.drain(..k));
        self.released += k;
    }

    /// Feed native-rate mono; append releasable output to `out`.
    pub fn push(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<()> {
        self.total_in += input.len();
        let mut rest = input;
        while !rest.is_empty() {
            if self.pending.is_empty() && rest.len() >= RESAMPLE_CHUNK {
                // Whole chunk available: process straight from the caller's slice.
                let (chunk, tail) = rest.split_at(RESAMPLE_CHUNK);
                self.run(chunk, None)?;
                rest = tail;
            } else {
                let take = (RESAMPLE_CHUNK - self.pending.len()).min(rest.len());
                self.pending.extend_from_slice(&rest[..take]);
                rest = &rest[take..];
                if self.pending.len() == RESAMPLE_CHUNK {
                    let chunk = std::mem::take(&mut self.pending);
                    self.run(&chunk, None)?;
                    self.pending = chunk;
                    self.pending.clear();
                }
            }
        }
        self.release(out);
        Ok(())
    }

    /// Flush and append the rest. Total appended over the stream equals the one-shot length.
    pub fn finish(mut self, out: &mut Vec<f32>) -> Result<()> {
        if self.total_in == 0 {
            bail!("cannot resample empty input");
        }
        let target = one_shot_len(self.total_in, self.ratio);
        let valid = self.pending.len();
        let mut chunk = std::mem::take(&mut self.pending);
        chunk.resize(RESAMPLE_CHUNK, 0.0);
        self.run(&chunk, Some(valid))?;
        chunk.fill(0.0);
        while self.released + self.produced.len() < target {
            self.run(&chunk, Some(0))?;
        }
        let k = target - self.released;
        out.extend(self.produced.drain(..k));
        Ok(())
    }
}

/// Resample mono audio from `source_sr` to `target_sr` using sinc interpolation.
/// Returns samples unchanged if rates already match.
///
/// Where chunked processing is provably bit-identical to the one-shot call
/// ([`chunking_is_exact`]: sources of 22050 * 2^k Hz, such as 44.1 kHz) the signal is resampled in
/// fixed chunks, so rubato's buffers stay small and the input is freed before the flush. Other
/// rates use the one-shot call, which is O(signal) in memory; see [`StreamResampler`] for the
/// chunked path at any rate.
pub fn resample(samples: Vec<f32>, source_sr: u32, target_sr: u32) -> Result<Vec<f32>> {
    if source_sr == target_sr {
        return Ok(samples);
    }
    if chunking_is_exact(source_sr, target_sr) {
        let mut resampler = StreamResampler::new(source_sr, target_sr)?;
        let mut out = Vec::with_capacity(one_shot_len(samples.len(), resampler.ratio));
        resampler.push(&samples, &mut out)?;
        drop(samples);
        resampler.finish(&mut out)?;
        return Ok(out);
    }
    resample_one_shot(samples, source_sr, target_sr)
}

/// The whole input as one rubato chunk.
fn resample_one_shot(samples: Vec<f32>, source_sr: u32, target_sr: u32) -> Result<Vec<f32>> {
    let params = sinc_params();

    // rubato 3.0: `SincFixedIn` became `Async` with `FixedAsync::Input`. We
    // process the whole buffer in a single call (chunk_size = input length),
    // matching the previous one-shot behavior.
    let frames = samples.len();
    let mut resampler = Async::<f32>::new_sinc(
        target_sr as f64 / source_sr as f64,
        2.0,
        &params,
        frames,
        1, // mono
        FixedAsync::Input,
    )?;

    // For mono, interleaved layout is just the flat sample slice.
    let input = InterleavedSlice::new(&samples, 1, frames)
        .map_err(|e| anyhow!("resampler input adapter: {e:?}"))?;
    let output = resampler.process(&input, 0, None)?;
    Ok(output.take_data())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_mono_passthrough() {
        let samples = vec![1.0, 2.0, 3.0];
        let mono = to_mono(&samples, 1);
        assert_eq!(mono, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_to_mono_stereo() {
        // Two frames of stereo: (0.5, 1.5) and (1.0, 3.0)
        let samples = vec![0.5, 1.5, 1.0, 3.0];
        let mono = to_mono(&samples, 2);
        assert_eq!(mono, vec![1.0, 2.0]);
    }

    #[test]
    fn test_resample_identity() {
        // Same rate should return unchanged
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let result = resample(samples.clone(), 22050, 22050).unwrap();
        assert_eq!(result, samples);
    }
}
