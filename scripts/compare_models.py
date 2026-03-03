# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Compare FP32 and Int8 beat models for accuracy and performance.

Runs both models on all integration test audio files, compares beat timestamps
using nearest-match pairing, and reports maximum deviation and timing.

Usage:
    uv run scripts/compare_models.py
    uv run scripts/compare_models.py --fp32 models/beat_this.onnx --int8 models/beat_this_int8.onnx
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "integration_test_files"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}
TOLERANCE = 0.020  # 20ms = 1 frame at 50 fps


def find_binary() -> Path:
    """Find the beat-this binary (release or debug)."""
    for profile in ("release", "debug"):
        path = ROOT / "target" / profile / "beat-this"
        if path.exists():
            return path
    sys.exit("beat-this binary not found. Run: cargo build --release")


def discover_audio_files() -> list[Path]:
    files = sorted(
        f for f in AUDIO_DIR.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        sys.exit(f"No audio files found in {AUDIO_DIR}")
    return files


def run_model(binary: Path, audio: Path, model: Path, output_json: Path) -> float | None:
    """Run beat-this with a given model. Returns elapsed seconds or None on failure."""
    cmd = [str(binary), str(audio), "--model", str(model)]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"    FAILED: {result.stderr.strip()}")
        return None

    output_json.write_text(result.stdout)
    return elapsed


def parse_beats(json_path: Path) -> tuple[list[float], list[float]]:
    """Parse JSON output. Returns (beat_times, downbeat_times)."""
    data = json.loads(json_path.read_text())
    beats = [b["time"] for b in data["beats"]]
    downbeats = data["downbeats"]
    return beats, downbeats


def nearest_match(ref: list[float], test: list[float], tolerance: float) -> dict:
    """Match each ref beat to its nearest test beat within tolerance.

    Returns stats about matched, missed (in ref but not test), and
    spurious (in test but not ref) beats.
    """
    matched = []
    deviations = []
    used_test = set()

    for r in ref:
        best_idx = None
        best_dev = float("inf")
        for j, t in enumerate(test):
            if j in used_test:
                continue
            dev = abs(r - t)
            if dev < best_dev:
                best_dev = dev
                best_idx = j
            # Since both lists are sorted, stop searching if we've passed
            if t > r + tolerance * 10:
                break
        if best_idx is not None and best_dev <= tolerance:
            matched.append((r, test[best_idx], best_dev))
            deviations.append(best_dev)
            used_test.add(best_idx)
        else:
            matched.append((r, None, None))

    missed = sum(1 for _, t, _ in matched if t is None)
    spurious = len(test) - len(used_test)
    max_dev = max(deviations) if deviations else 0.0

    return {
        "ref_count": len(ref),
        "test_count": len(test),
        "matched": len(deviations),
        "missed": missed,
        "spurious": spurious,
        "max_deviation_ms": max_dev * 1000,
        "mean_deviation_ms": (sum(deviations) / len(deviations) * 1000) if deviations else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare FP32 and Int8 models")
    parser.add_argument(
        "--fp32",
        type=Path,
        default=ROOT / "models" / "beat_this.onnx",
    )
    parser.add_argument(
        "--int8",
        type=Path,
        default=ROOT / "models" / "beat_this_int8.onnx",
    )
    args = parser.parse_args()

    for model in (args.fp32, args.int8):
        if not model.exists():
            sys.exit(f"Model not found: {model}")

    binary = find_binary()
    audio_files = discover_audio_files()

    print(f"Binary:      {binary}")
    print(f"FP32 model:  {args.fp32} ({args.fp32.stat().st_size / 1e6:.1f} MB)")
    print(f"Int8 model:  {args.int8} ({args.int8.stat().st_size / 1e6:.1f} MB)")
    print(f"Audio files: {len(audio_files)}")
    print(f"Tolerance:   {TOLERANCE * 1000:.0f}ms")
    print()

    all_pass = True
    overall_max_dev = 0.0
    fp32_times = []
    int8_times = []
    file_results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for audio in audio_files:
            rel = audio.relative_to(ROOT)
            print(f"--- {rel} ---")

            fp32_json = tmpdir / f"{audio.stem}_fp32.json"
            int8_json = tmpdir / f"{audio.stem}_int8.json"

            # Run FP32
            print("  FP32...", end=" ", flush=True)
            fp32_time = run_model(binary, audio, args.fp32, fp32_json)
            if fp32_time is not None:
                print(f"{fp32_time:.2f}s")
                fp32_times.append(fp32_time)
            else:
                print("FAILED")
                all_pass = False
                continue

            # Run Int8
            print("  Int8...", end=" ", flush=True)
            int8_time = run_model(binary, audio, args.int8, int8_json)
            if int8_time is not None:
                print(f"{int8_time:.2f}s")
                int8_times.append(int8_time)
            else:
                print("FAILED")
                all_pass = False
                continue

            # Compare beats
            fp32_beats, fp32_db = parse_beats(fp32_json)
            int8_beats, int8_db = parse_beats(int8_json)
            beat_stats = nearest_match(fp32_beats, int8_beats, TOLERANCE)
            db_stats = nearest_match(fp32_db, int8_db, TOLERANCE)

            overall_max_dev = max(overall_max_dev, beat_stats["max_deviation_ms"])
            speedup = fp32_time / int8_time if int8_time > 0 else 0

            passed = (
                beat_stats["missed"] == 0
                and beat_stats["spurious"] == 0
                and beat_stats["max_deviation_ms"] <= TOLERANCE * 1000
            )
            if not passed:
                all_pass = False

            status = "PASS" if passed else "FAIL"
            print(f"  {status}: {beat_stats['ref_count']} beats "
                  f"({beat_stats['matched']} matched, "
                  f"{beat_stats['missed']} missed, "
                  f"{beat_stats['spurious']} spurious)")
            print(f"    Beat deviation:     max={beat_stats['max_deviation_ms']:.1f}ms, "
                  f"mean={beat_stats['mean_deviation_ms']:.1f}ms")
            print(f"    Downbeat deviation: max={db_stats['max_deviation_ms']:.1f}ms "
                  f"({db_stats['matched']}/{db_stats['ref_count']} matched, "
                  f"{db_stats['missed']} missed, {db_stats['spurious']} spurious)")
            print(f"    Speedup: {speedup:.2f}x")
            print()

            file_results.append({
                "file": str(rel),
                "fp32_time": fp32_time,
                "int8_time": int8_time,
                "speedup": speedup,
                "beats": beat_stats,
                "downbeats": db_stats,
                "passed": passed,
            })

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Overall max beat deviation: {overall_max_dev:.1f}ms (tolerance: {TOLERANCE * 1000:.0f}ms)")
    print(f"Result: {'PASS' if all_pass else 'FAIL'}")

    if fp32_times and int8_times:
        fp32_total = sum(fp32_times)
        int8_total = sum(int8_times)
        speedup = fp32_total / int8_total if int8_total > 0 else 0
        print(f"\nFP32 total: {fp32_total:.2f}s")
        print(f"Int8 total: {int8_total:.2f}s")
        print(f"Speedup:    {speedup:.2f}x")

    # Write detailed results as JSON
    results_path = ROOT / "docs" / "tasks" / "014-next-steps" / "int8q-comparison.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(file_results, indent=2) + "\n")
    print(f"\nDetailed results: {results_path.relative_to(ROOT)}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
