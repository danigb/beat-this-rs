# Plan: Docker Image for beat-this-rs

## Deliverables

1. `Dockerfile` — multi-stage build (rust builder + debian runtime)
2. `.dockerignore` — keep build context small
3. Verify it builds and runs locally on macOS

## Step 1: Create `.dockerignore`

Exclude everything not needed for compilation or the final image. Models are handled separately via COPY of specific files.

```
.git/
target/
references/
docs/
scripts/
tests/
integration_test_files/
tmp/
.claude/
.vscode/
.DS_Store
*.wav
*.beats
*.ckpt
onnxruntime
```

## Step 2: Create `Dockerfile`

Multi-stage build with Option A (COPY models from local context):

```dockerfile
# Stage 1: Build the release binary
FROM rust:bookworm AS builder

RUN apt-get update && apt-get install -y pkg-config libssl-dev

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/

RUN cargo build --release

# Stage 2: Runtime image
FROM debian:bookworm

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/beat-this /usr/local/bin/beat-this

# Models: copy from local build context (Option A)
COPY models/mel_spectrogram.onnx /app/models/
COPY models/beat_this.onnx /app/models/

WORKDIR /app
ENTRYPOINT ["beat-this"]
CMD ["--help"]
```

Notes:
- `CMD ["--help"]` so running with no args shows usage instead of an error.
- Models default paths are `models/beat_this.onnx` and `models/mel_spectrogram.onnx`. Since `WORKDIR` is `/app` and models are at `/app/models/`, the defaults work without extra flags.
- Runtime is `rten` by default — no extra libraries needed.

## Step 3: Build and test locally

Build:

```bash
docker build -t beat-this .
```

Test with a local audio file:

```bash
docker run --rm -v $(pwd)/testdata:/audio beat-this /audio/test.wav
```

Verify JSON output on stdout, check exit code.

Test with output files:

```bash
docker run --rm -v $(pwd)/testdata:/audio beat-this /audio/test.wav --output-click /audio/clicks.wav
```

Verify the WAV file is written to the mounted volume.

## Step 4: Test cross-platform (optional)

If the target server is x86_64 and building on Apple Silicon:

```bash
docker buildx build --platform linux/amd64 -t beat-this-amd64 .
docker run --rm --platform linux/amd64 -v $(pwd)/testdata:/audio beat-this-amd64 /audio/test.wav
```

## Out of scope (for now)

- Hosting models on GitHub Releases (Option B from research)
- Multi-arch image publishing
- CI/CD pipeline for automated image builds
- `target-cpu=native` optimization (requires building on the same arch as the server)
