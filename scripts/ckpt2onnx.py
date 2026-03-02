# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0",
#     "onnx",
#     "onnxscript",
#     "beat-this @ https://github.com/CPJKU/beat_this/archive/main.zip",
# ]
# ///
"""Download a Beat This! checkpoint and convert it to ONNX format.

Usage:
    uv run scripts/ckpt2onnx.py final0
    uv run scripts/ckpt2onnx.py small0
"""

import sys
import inspect
from pathlib import Path
from urllib.request import urlretrieve

import torch

from beat_this.model.beat_tracker import BeatThis
from beat_this.utils import replace_state_dict_key

CHECKPOINT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class BeatThisWrapper(torch.nn.Module):
    """Thin wrapper that returns a tuple instead of a dict (ONNX requirement)."""

    def __init__(self, model: BeatThis):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return out["beat"], out["downbeat"]


def download(name: str) -> Path:
    """Download a checkpoint from the cloud space into models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / f"{name}.ckpt"
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return dest
    url = f"{CHECKPOINT_URL}/{name}.ckpt"
    print(f"Downloading {url} ...")
    urlretrieve(url, dest)
    print(f"Saved: {dest}")
    return dest


def convert(ckpt_path: Path) -> None:
    """Convert a .ckpt checkpoint to .onnx in the same directory."""
    onnx_path = ckpt_path.with_suffix(".onnx")

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Extract hyperparameters applicable to the model
    hparams = checkpoint["hyper_parameters"]
    valid_params = set(inspect.signature(BeatThis).parameters)
    hparams = {k: v for k, v in hparams.items() if k in valid_params}

    # Build the model and load weights
    model = BeatThis(**hparams)
    state_dict = replace_state_dict_key(checkpoint["state_dict"], "model.", "")
    model.load_state_dict(state_dict)
    model.eval()

    wrapper = BeatThisWrapper(model)
    wrapper.eval()

    # Dummy input: (batch=1, time_frames=1500, mel_bins=128)
    dummy = torch.randn(1, 1500, 128)

    print(f"Exporting to: {onnx_path}")
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=["spectrogram"],
        output_names=["beat", "downbeat"],
        dynamic_axes={
            "spectrogram": {0: "batch", 1: "time"},
            "beat": {0: "batch", 1: "time"},
            "downbeat": {0: "batch", 1: "time"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Done: {onnx_path}")


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: uv run scripts/ckpt2onnx.py <model_name>")
    name = sys.argv[1]
    ckpt_path = download(name)
    convert(ckpt_path)


if __name__ == "__main__":
    main()
