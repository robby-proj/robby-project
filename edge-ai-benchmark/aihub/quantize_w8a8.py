#!/usr/bin/env python3
"""Submit W8A8 AI Hub quantization using a COCO-style calibration image folder."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import qai_hub as hub


def letterbox_rgb_tensor(path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not read calibration image: {path}")
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_width = round(width * scale)
    new_height = round(height * scale)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_width = size - new_width
    pad_height = size - new_height
    left = round(pad_width / 2 - 0.1)
    top = round(pad_height / 2 - 0.1)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + new_height, left : left + new_width] = image
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="AI Hub ID of split-output FP32 model")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--name", default="VentunoQ-YOLOv8n-W8A8")
    args = parser.parse_args()

    paths = sorted(args.images_dir.glob("*.jpg"))[:: args.stride][: args.limit]
    if not paths:
        raise RuntimeError(f"No JPG files found in {args.images_dir}")
    samples = [letterbox_rgb_tensor(path, args.size) for path in paths]
    print(f"Calibration tensors: {len(samples)}, shape={samples[0].shape}, dtype={samples[0].dtype}")

    job = hub.submit_quantize_job(
        model=hub.get_model(args.model_id),
        calibration_data={"images": samples},
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
        name=args.name,
    )
    print("Quantization job:", job)
    job.wait()
    model = job.get_target_model()
    print("Quantized model ID:", model.model_id)


if __name__ == "__main__":
    main()
