# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "einops==0.8.0",
#     "numpy==1.26.4",
#     "rotary_embedding_torch==0.6.4",
#     "soundfile",
#     "soxr==0.3.7",
#     "torch==2.3.1",
#     "torchaudio==2.3.1",
#     "tqdm==4.66.4",
# ]
# ///
"""
Integration test: compare Python and Rust beat-this implementations.

Builds the Rust binary, runs both versions on all audio files in
integration_test_files/, compares outputs, and reports timing.

Usage:
    uv run scripts/integration-test.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "integration_test_files"
RUST_BINARY = ROOT / "target" / "release" / "beat-this"
PYTHON_CLI = ROOT / "references" / "beat_this" / "beat_this" / "cli.py"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}
TIMESTAMP_TOLERANCE = 0.02  # seconds


def build_rust():
    print("=== Building Rust binary (release) ===")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("FAILED to build Rust binary:")
        print(result.stderr)
        sys.exit(1)
    print("Build OK\n")


def discover_audio_files():
    files = sorted(
        f
        for f in AUDIO_DIR.rglob("*")
        if f.suffix.lower() in AUDIO_EXTENSIONS
    )
    print(f"=== Found {len(files)} audio files ===")
    for f in files:
        print(f"  {f.relative_to(ROOT)}")
    print()
    return files


def run_python(audio_path: Path, output_path: Path):
    """Run Python beat_this and return elapsed seconds."""
    env = {
        "PYTHONPATH": str(ROOT / "references" / "beat_this"),
        "PATH": subprocess.os.environ.get("PATH", ""),
    }
    # Inherit common env vars needed for Python/torch
    for key in ("HOME", "USER", "VIRTUAL_ENV", "CONDA_PREFIX", "CUDA_VISIBLE_DEVICES"):
        if key in subprocess.os.environ:
            env[key] = subprocess.os.environ[key]

    cmd = [
        sys.executable,
        str(PYTHON_CLI),
        str(audio_path),
        "-o", str(output_path),
        "--no-dbn",
        "--gpu", "-1",
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"    Python FAILED: {result.stderr.strip()}")
        return None

    return elapsed


def run_rust(audio_path: Path, output_path: Path):
    """Run Rust beat-this and return elapsed seconds."""
    cmd = [
        str(RUST_BINARY),
        str(audio_path),
        "--output-beats",
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"    Rust FAILED: {result.stderr.strip()}")
        return None

    output_path.write_text(result.stdout)
    return elapsed


def parse_beats(path: Path):
    """Parse a .beats file into list of (time, beat) tuples."""
    beats = []
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2:
            beats.append((float(parts[0]), int(parts[1])))
    return beats


def compare_beats(python_beats, rust_beats):
    """Compare two beat lists. Returns list of difference descriptions."""
    diffs = []

    len_py = len(python_beats)
    len_rs = len(rust_beats)
    if len_py != len_rs:
        diffs.append(f"Beat count differs: python={len_py}, rust={len_rs}")

    # Compare aligned entries
    for i in range(min(len_py, len_rs)):
        py_time, py_beat = python_beats[i]
        rs_time, rs_beat = rust_beats[i]

        time_diff = abs(py_time - rs_time)
        if time_diff > TIMESTAMP_TOLERANCE or py_beat != rs_beat:
            diffs.append(
                f"  [{i}] python=({py_time:.4f}, {py_beat}) "
                f"rust=({rs_time:.4f}, {rs_beat}) "
                f"time_diff={time_diff:.4f}s"
            )

    # Report extra entries
    if len_py > len_rs:
        for i in range(len_rs, len_py):
            diffs.append(f"  [{i}] python=({python_beats[i][0]:.4f}, {python_beats[i][1]}) rust=MISSING")
    elif len_rs > len_py:
        for i in range(len_py, len_rs):
            diffs.append(f"  [{i}] python=MISSING rust=({rust_beats[i][0]:.4f}, {rust_beats[i][1]})")

    return diffs


def process_file(audio_path: Path):
    """Process a single audio file. Returns (has_diffs, py_time, rs_time)."""
    stem = audio_path.with_suffix("")
    py_beats_path = Path(str(stem) + ".python.beats")
    rs_beats_path = Path(str(stem) + ".rust.beats")
    py_time_path = Path(str(stem) + ".python-time.txt")
    rs_time_path = Path(str(stem) + ".rust-time.txt")
    diff_path = Path(str(stem) + ".differences.txt")

    rel = audio_path.relative_to(ROOT)
    print(f"--- {rel} ---")

    # Run Python
    print("  Python...", end=" ", flush=True)
    py_time = run_python(audio_path, py_beats_path)
    if py_time is not None:
        py_time_path.write_text(f"{py_time:.3f}s\n")
        print(f"{py_time:.1f}s")
    else:
        print("FAILED")

    # Run Rust
    print("  Rust...  ", end=" ", flush=True)
    rs_time = run_rust(audio_path, rs_beats_path)
    if rs_time is not None:
        rs_time_path.write_text(f"{rs_time:.3f}s\n")
        print(f"{rs_time:.1f}s")
    else:
        print("FAILED")

    # Compare
    has_diffs = False
    if py_time is not None and rs_time is not None:
        python_beats = parse_beats(py_beats_path)
        rust_beats = parse_beats(rs_beats_path)
        diffs = compare_beats(python_beats, rust_beats)

        if diffs:
            has_diffs = True
            diff_text = f"Differences for {rel}\n"
            diff_text += f"Python beats: {len(python_beats)}, Rust beats: {len(rust_beats)}\n"
            diff_text += f"Tolerance: {TIMESTAMP_TOLERANCE}s\n\n"
            diff_text += "\n".join(diffs) + "\n"
            diff_path.write_text(diff_text)
            print(f"  DIFFERENCES: {len(diffs)} (see {diff_path.name})")
        else:
            # Remove stale diff file if exists
            diff_path.unlink(missing_ok=True)
            print(f"  MATCH ({len(rust_beats)} beats)")
    else:
        print("  Comparison skipped (one or both failed)")

    return has_diffs, py_time, rs_time


def main():
    build_rust()
    audio_files = discover_audio_files()

    if not audio_files:
        print("No audio files found in integration_test_files/")
        sys.exit(1)

    matches = 0
    diffs = 0
    failures = 0
    py_times = []
    rs_times = []

    print("=== Running integration tests ===\n")

    for audio_path in audio_files:
        has_diffs, py_time, rs_time = process_file(audio_path)
        print()

        if py_time is None or rs_time is None:
            failures += 1
        elif has_diffs:
            diffs += 1
        else:
            matches += 1

        if py_time is not None:
            py_times.append(py_time)
        if rs_time is not None:
            rs_times.append(rs_time)

    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Files processed: {len(audio_files)}")
    print(f"  Matching:      {matches}")
    print(f"  Differences:   {diffs}")
    print(f"  Failures:      {failures}")

    if py_times and rs_times:
        py_total = sum(py_times)
        rs_total = sum(rs_times)
        print(f"\nPython total time: {py_total:.1f}s (avg {py_total / len(py_times):.1f}s)")
        print(f"Rust total time:   {rs_total:.1f}s (avg {rs_total / len(rs_times):.1f}s)")
        if rs_total > 0:
            print(f"Speedup: {py_total / rs_total:.2f}x")

    # Save report JSON
    report = {
        "files_processed": len(audio_files),
        "matching": matches,
        "differences": diffs,
        "failures": failures,
        "files": [],
    }
    for audio_path, py_time, rs_time in zip(audio_files, py_times, rs_times):
        entry = {
            "file": str(audio_path.relative_to(ROOT)),
            "python_time": round(py_time, 3) if py_time is not None else None,
            "rust_time": round(rs_time, 3) if rs_time is not None else None,
        }
        if py_time and rs_time:
            entry["speedup"] = round(py_time / rs_time, 2)
        report["files"].append(entry)
    if py_times and rs_times:
        report["python_total"] = round(sum(py_times), 3)
        report["rust_total"] = round(sum(rs_times), 3)
        report["speedup"] = round(sum(py_times) / sum(rs_times), 2)

    report_path = AUDIO_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nReport saved to {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
