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

COPY models/mel_spectrogram.onnx /app/models/
COPY models/beat_this.onnx /app/models/

WORKDIR /app
ENTRYPOINT ["beat-this"]
CMD ["--help"]
