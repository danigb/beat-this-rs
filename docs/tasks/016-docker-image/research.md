# Research: Docker Image for beat-this-rs

## Priority: Maximum Compatibility and Performance

## 1. System Dependencies

### Build Time

- Rust toolchain
- `pkg-config`, `gcc` (C linker)
- `libssl-dev` (pulled transitively by `ort-sys` during compilation)

### Runtime

The default `rten` backend is pure Rust — **zero runtime system dependencies**. All audio decoding (symphonia), resampling (rubato), WAV writing (hound), and inference (rten) compile into the binary.

The `ort` backend requires `libonnxruntime.so` at runtime (loaded via `dlopen`), but its main advantage is CoreML on macOS, which is irrelevant in a Linux container. **Use `rten` in Docker.**

## 2. Base Image: `rust:bookworm` / `debian:bookworm`

Use full Debian Bookworm (not slim, not Alpine):

- **glibc**: rten uses SIMD intrinsics and Rayon thread pool. glibc is the best-tested target for numerical workloads in Rust. musl/Alpine may have performance regressions.
- **Compatibility**: Bookworm is the current stable Debian. Broad hardware and library support.
- **Debugging**: Full image includes common tools (`bash`, `ls`, `curl`) useful for troubleshooting in production.

## 3. Performance Considerations

- **CPU cores**: rten parallelizes via Rayon. More cores = faster inference. On multi-core, inference is ~4.6x faster than single-threaded. Don't limit `--cpus` in Docker unless necessary.
- **Memory**: Standard model needs ~200 MB working memory. Not a concern for most servers.
- **Release build**: Must use `cargo build --release`. Debug builds are 10-50x slower for inference.
- **Native CPU features**: Consider `RUSTFLAGS="-C target-cpu=native"` if building and running on the same machine for maximum SIMD optimization. For portable images, omit this flag (the default target is safe for any x86_64/arm64).

## 4. Model Files Strategy

Models are gitignored and live locally in `models/`. The required files are:

| File | Size | Required |
|------|------|----------|
| `mel_spectrogram.onnx` | 268 KB | Always |
| `beat_this.onnx` | 79 MB | Standard model |

There are also variant models (`beat_this_small.onnx` at 10 MB, `beat_this_int8.onnx` at 23 MB) but the standard model should be the default for maximum accuracy.

Models are originally obtained by downloading checkpoints from `https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/` and converting with `scripts/ckpt2onnx.py` (requires Python + PyTorch).

### Option A: COPY from local build context

The Dockerfile copies models from the host filesystem during `docker build`.

```dockerfile
COPY models/mel_spectrogram.onnx /app/models/
COPY models/beat_this.onnx /app/models/
```

| Pros | Cons |
|------|------|
| Simplest approach | Models must exist locally before `docker build` |
| Fast builds (no download) | Anyone building the image needs the models first |
| Works offline | Build context upload includes ~80 MB of model data |
| No external service dependency | Not self-contained — can't just clone + build |

**Best for**: Personal use, deploying from a dev machine that already has the models.

### Option B: Download pre-converted ONNX from a URL during build

Add a `RUN curl` step in the Dockerfile to fetch the ONNX files from a hosted location (e.g., GitHub Releases, cloud storage, or the original JKU server).

```dockerfile
RUN curl -L -o /app/models/beat_this.onnx \
    "https://github.com/danigb/beat-this-rs/releases/download/v0.1.0/beat_this.onnx"
```

The JKU server only hosts `.ckpt` checkpoints (not ONNX), so you'd need to host the pre-converted ONNX files yourself.

| Pros | Cons |
|------|------|
| Self-contained Dockerfile | Need to host ONNX files somewhere |
| Reproducible builds | GitHub Releases has 2 GB file limit (fine for 79 MB) |
| Anyone can build without local models | Build requires network access |
| Versioned if using release tags | Hosting could go down |

**Best for**: Shareable Dockerfile, CI/CD pipelines, open-source distribution.

### Option C: Volume mount at runtime (no models in image)

Don't bake models into the image. The user mounts a local directory containing models when running the container.

```bash
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/audio:/audio \
  beat-this /audio/song.wav
```

```dockerfile
# No COPY for models — they come from a volume
WORKDIR /app
ENTRYPOINT ["beat-this", "--model", "/app/models/beat_this.onnx", "--mel-model", "/app/models/mel_spectrogram.onnx"]
```

| Pros | Cons |
|------|------|
| Smallest image (~90 MB less) | More complex `docker run` command |
| Flexible — swap models without rebuilding | User must manage models separately |
| Same image works with any model variant | Easy to forget the mount and get a confusing error |
| Models can be shared across containers | |

**Best for**: Servers that process many files with different model variants, or when image size on the registry matters.

### Option D: Download checkpoint + convert during build

Run the Python conversion script inside the Docker build. Requires a Python + PyTorch layer.

```dockerfile
FROM python:3.11 AS converter
RUN pip install torch onnx beat-this
COPY scripts/ckpt2onnx.py /scripts/
RUN python /scripts/ckpt2onnx.py final0
# then copy the .onnx output to the runtime stage
```

| Pros | Cons |
|------|------|
| Fully reproducible from source | Enormous build image (PyTorch is ~2 GB) |
| No need to host ONNX files | Very slow builds (download + install + convert) |
| Always gets the latest checkpoint | Three-stage build (python, rust, runtime) |

**Best for**: Probably not worth it. Only makes sense if model provenance from source is critical.

### Recommendation

**Start with Option A** (COPY from local context). It's the simplest path to get running. If you later want to share the image or build in CI, move to **Option B** by uploading the ONNX files to a GitHub Release.

**Option C** (volume mount) is a good complement — you can support both by making the model paths configurable with defaults that work when models are baked in, and overridable via `--model`/`--mel-model` when mounted.

## 5. Proposed Dockerfile

```dockerfile
# Stage 1: Build
FROM rust:bookworm AS builder

RUN apt-get update && apt-get install -y pkg-config libssl-dev

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/

RUN cargo build --release

# Stage 2: Runtime
FROM debian:bookworm

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/beat-this /usr/local/bin/beat-this
COPY models/mel_spectrogram.onnx /app/models/
COPY models/beat_this.onnx /app/models/

WORKDIR /app
ENTRYPOINT ["beat-this"]
```

## 6. Testing Locally on macOS

### Build the image

```bash
docker build -t beat-this .
```

On Apple Silicon, Docker Desktop builds for `linux/arm64` by default, which is fine — it runs natively without emulation.

### Run on a single file

```bash
docker run --rm -v $(pwd)/testdata:/audio beat-this /audio/input.wav
```

### Run with output files

```bash
docker run --rm -v $(pwd):/data beat-this /data/song.wav --output-click /data/clicks.wav
```

### Test linux/amd64 (to match a typical server)

```bash
docker buildx build --platform linux/amd64 -t beat-this-amd64 .
docker run --rm --platform linux/amd64 -v $(pwd)/testdata:/audio beat-this-amd64 /audio/input.wav
```

## 7. Deploying to a Server

### Push to registry

```bash
docker tag beat-this ghcr.io/danigb/beat-this:latest
docker push ghcr.io/danigb/beat-this:latest
```

### Pull and run on server

```bash
docker pull ghcr.io/danigb/beat-this:latest

# Single file
docker run --rm -v /path/to/audio:/audio ghcr.io/danigb/beat-this /audio/song.wav

# Batch processing
docker run --rm -v /path/to/music:/audio ghcr.io/danigb/beat-this /audio -r
```

## 8. Open Questions

1. **Target architecture**: Is the server x86_64 or arm64? This determines whether to build a multi-arch image or just one platform.
2. **Model hosting**: If moving to Option B, where to host the ONNX files — GitHub Releases is the easiest.
3. **`target-cpu=native`**: Worth it if building on the same architecture as the server. Skippable for portable images.
