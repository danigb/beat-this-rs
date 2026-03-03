# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "onnxruntime",
#     "onnx",
# ]
# ///
"""Quantize the Beat This! ONNX model from FP32 to Int8 (dynamic quantization).

Produces a quantized model that is ~4x smaller and 2-3x faster on CPUs with
integer instruction support (ARM NEON, x86 VNNI).

Usage:
    uv run scripts/quantize_int8.py
    uv run scripts/quantize_int8.py --input models/beat_this.onnx --output models/beat_this_int8.onnx
"""

import argparse
from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, QuantType

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    parser = argparse.ArgumentParser(description="Quantize Beat This! model to Int8")
    parser.add_argument(
        "--input",
        type=Path,
        default=MODELS_DIR / "beat_this.onnx",
        help="Input FP32 ONNX model (default: models/beat_this.onnx)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODELS_DIR / "beat_this_int8.onnx",
        help="Output Int8 ONNX model (default: models/beat_this_int8.onnx)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input model not found: {args.input}")

    input_size = args.input.stat().st_size / (1024 * 1024)
    print(f"Input model:  {args.input} ({input_size:.1f} MB)")
    print(f"Output model: {args.output}")
    print()

    print("Quantizing (dynamic Int8, targeting MatMul ops)...")
    quantize_dynamic(
        model_input=str(args.input),
        model_output=str(args.output),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
    )

    output_size = args.output.stat().st_size / (1024 * 1024)
    ratio = input_size / output_size
    print(f"\nDone: {args.output} ({output_size:.1f} MB, {ratio:.1f}x smaller)")


if __name__ == "__main__":
    main()
