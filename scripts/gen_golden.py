# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0",
#     "soxr",
#     "soundfile",
#     "beat-this @ https://github.com/CPJKU/beat_this/archive/b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c.zip",
# ]
# ///
"""Maintainer tool: generate the Python-reference golden beats/downbeats for the
parity test (tests/python_parity.rs).

End users do NOT need this — the goldens are committed under tests/fixtures/.
Use this only to (re)generate them if the checkpoint or the mel/inference graph
changes.

The <checkpoint> argument may be a local .ckpt path (e.g. models/small1.ckpt) or
a beat_this shortname (e.g. small1, final0). Shortnames download from the cloud
space; local paths avoid the network.

Usage:
    uv run scripts/gen_golden.py models/small1.ckpt tests/fixtures/golden_small.json
    uv run scripts/gen_golden.py models/final0.ckpt tests/fixtures/golden_full.json
"""

import json
import sys
from pathlib import Path

from beat_this.inference import File2Beats

ROOT = Path(__file__).resolve().parent.parent
MP3 = ROOT / "test_files" / "It Don't Mean A Thing - Kings of Swing.mp3"

# Frozen beat_this reference the goldens are built from. Keep this in sync with the
# pinned commit in the `# /// script` dependency URL above. uv fetches the package
# from a zip (not a git checkout), so we record the pinned commit directly rather
# than deriving it via `git rev-parse`.
BEAT_THIS_COMMIT = "b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c"


def beat_this_version() -> str:
    try:
        from importlib.metadata import version

        return version("beat-this")
    except Exception:
        return "unknown"


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: uv run scripts/gen_golden.py <checkpoint> <out.json>")
    checkpoint, out = sys.argv[1], Path(sys.argv[2])

    f2b = File2Beats(checkpoint_path=checkpoint, device="cpu", dbn=False)
    beats, downbeats = f2b(MP3)  # (beats, downbeats) in seconds

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "beats": [round(float(t), 6) for t in beats],
                "downbeats": [round(float(t), 6) for t in downbeats],
                "provenance": {
                    "checkpoint": checkpoint,
                    "audio": MP3.name,
                    "beat_this_version": beat_this_version(),
                    "beat_this_commit": BEAT_THIS_COMMIT,
                    "postprocessing": "minimal",
                    "fps": 50,
                    "command": f"uv run scripts/gen_golden.py {checkpoint} {out}",
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {len(beats)} beats, {len(downbeats)} downbeats -> {out}")


if __name__ == "__main__":
    main()
