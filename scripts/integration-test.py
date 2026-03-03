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

Builds the Rust binary, runs both Rust runtimes (rten and ort) against the
Python reference on all audio files in integration_test_files/, compares
outputs, and reports timing.

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
RUST_RUNTIMES = ["rten", "ort"]


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


def find_onnxruntime_lib_dir():
    """Find ONNX Runtime library directory for dynamic loading."""
    candidates = [
        Path("/opt/homebrew/lib"),
        Path("/usr/local/lib"),
    ]
    for d in candidates:
        if (d / "libonnxruntime.dylib").exists():
            return str(d)
    return None


ORT_LIB_DIR = find_onnxruntime_lib_dir()


def run_rust(audio_path: Path, output_path: Path, runtime: str):
    """Run Rust beat-this with given runtime and return elapsed seconds."""
    cmd = [
        str(RUST_BINARY),
        str(audio_path),
        "--runtime", runtime,
        f"--beats={output_path}",
        "--overwrite",
    ]
    env = dict(subprocess.os.environ)
    if runtime == "ort" and ORT_LIB_DIR:
        env["DYLD_LIBRARY_PATH"] = ORT_LIB_DIR
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"    Rust ({runtime}) FAILED: {result.stderr.strip()}")
        return None

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
    """Process a single audio file with all runtimes.

    Returns (file_results dict with times and diff counts).
    """
    stem = audio_path.with_suffix("")
    rel = audio_path.relative_to(ROOT)
    print(f"--- {rel} ---")

    file_result = {"file": str(rel)}

    # Run Python
    py_beats_path = Path(str(stem) + ".python.beats")
    print("  Python...", end=" ", flush=True)
    py_time = run_python(audio_path, py_beats_path)
    if py_time is not None:
        print(f"{py_time:.1f}s")
        file_result["python_time"] = round(py_time, 3)
    else:
        print("FAILED")
        file_result["python_time"] = None

    # Run each Rust runtime
    for runtime in RUST_RUNTIMES:
        rs_beats_path = Path(f"{stem}.rust-{runtime}.beats")
        diff_path = Path(f"{stem}.rust-{runtime}.differences.txt")

        print(f"  Rust ({runtime})...", end=" ", flush=True)
        rs_time = run_rust(audio_path, rs_beats_path, runtime)

        if rs_time is not None:
            print(f"{rs_time:.1f}s")
            file_result[f"{runtime}_time"] = round(rs_time, 3)
        else:
            print("FAILED")
            file_result[f"{runtime}_time"] = None
            continue

        # Compare against Python
        if py_time is not None:
            python_beats = parse_beats(py_beats_path)
            rust_beats = parse_beats(rs_beats_path)
            diffs = compare_beats(python_beats, rust_beats)

            if diffs:
                diff_text = f"Differences for {rel} (runtime: {runtime})\n"
                diff_text += f"Python beats: {len(python_beats)}, Rust beats: {len(rust_beats)}\n"
                diff_text += f"Tolerance: {TIMESTAMP_TOLERANCE}s\n\n"
                diff_text += "\n".join(diffs) + "\n"
                diff_path.write_text(diff_text)
                print(f"    vs python: {len(diffs)} diffs (see {diff_path.name})")
                file_result[f"{runtime}_diffs"] = len(diffs)
            else:
                diff_path.unlink(missing_ok=True)
                print(f"    vs python: MATCH ({len(rust_beats)} beats)")
                file_result[f"{runtime}_diffs"] = 0

    print()
    return file_result


def main():
    build_rust()
    audio_files = discover_audio_files()

    if not audio_files:
        print("No audio files found in integration_test_files/")
        sys.exit(1)

    print("=== Running integration tests ===\n")

    file_results = []
    for audio_path in audio_files:
        file_results.append(process_file(audio_path))

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files processed: {len(audio_files)}\n")

    # Collect times per runtime
    all_times = {"python": []}
    for rt in RUST_RUNTIMES:
        all_times[rt] = []

    for fr in file_results:
        if fr.get("python_time") is not None:
            all_times["python"].append(fr["python_time"])
        for rt in RUST_RUNTIMES:
            if fr.get(f"{rt}_time") is not None:
                all_times[rt].append(fr[f"{rt}_time"])

    # Print timing table
    header = f"{'File':<30} {'Python':>8}"
    for rt in RUST_RUNTIMES:
        header += f" {rt:>8}"
    print(header)
    print("-" * len(header))

    for fr in file_results:
        name = Path(fr["file"]).stem
        row = f"{name:<30}"
        py_t = fr.get("python_time")
        row += f" {py_t:>7.1f}s" if py_t else f" {'FAIL':>8}"
        for rt in RUST_RUNTIMES:
            rt_t = fr.get(f"{rt}_time")
            row += f" {rt_t:>7.1f}s" if rt_t else f" {'FAIL':>8}"
        print(row)

    print("-" * len(header))

    py_total = sum(all_times["python"])
    row = f"{'TOTAL':<30} {py_total:>7.1f}s"
    for rt in RUST_RUNTIMES:
        rt_total = sum(all_times[rt])
        row += f" {rt_total:>7.1f}s"
    print(row)

    # Speedups
    print()
    for rt in RUST_RUNTIMES:
        rt_total = sum(all_times[rt])
        if rt_total > 0 and py_total > 0:
            print(f"Speedup vs Python ({rt}): {py_total / rt_total:.2f}x")

    # Save report JSON
    report = {
        "files_processed": len(audio_files),
        "runtimes": RUST_RUNTIMES,
        "files": file_results,
    }
    if py_total > 0:
        report["python_total"] = round(py_total, 3)
    for rt in RUST_RUNTIMES:
        rt_total = sum(all_times[rt])
        if rt_total > 0:
            report[f"{rt}_total"] = round(rt_total, 3)
            if py_total > 0:
                report[f"{rt}_speedup"] = round(py_total / rt_total, 2)

    report_path = AUDIO_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nReport saved to {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
