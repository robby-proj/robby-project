#!/usr/bin/env python3
"""Export an Ultralytics YOLO checkpoint to a static 640x640 ONNX model."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path, help="Ultralytics .pt checkpoint")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    exported = model.export(
        format="onnx",
        imgsz=args.image_size,
        batch=1,
        dynamic=False,
        simplify=False,
        opset=args.opset,
        project=str(args.output_dir) if args.output_dir else None,
    )
    print("Exported ONNX model:", exported)


if __name__ == "__main__":
    main()
