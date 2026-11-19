#!/usr/bin/env python3
"""Compile an ONNX file or existing AI Hub model for Arduino Ventuno Q."""

import argparse
from pathlib import Path

import qai_hub as hub


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--onnx", type=Path, help="Local ONNX model to upload")
    source.add_argument("--model-id", help="Existing Qualcomm AI Hub model ID")
    parser.add_argument("--output", type=Path, required=True, help="Target package ZIP path")
    parser.add_argument("--name", default="VentunoQ-YOLOv8n-compile")
    args = parser.parse_args()

    model = hub.get_model(args.model_id) if args.model_id else str(args.onnx)
    device = hub.Device(name="Arduino VENTUNO Q", os="24.04")
    job = hub.submit_compile_job(
        model=model,
        device=device,
        options="--target_runtime onnx",
        name=args.name,
    )
    print("Compile job:", job)
    print("Uploaded/source model ID:", getattr(job, "model_id", "not reported"))
    job.wait()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    job.download_target_model(str(args.output))
    print("Downloaded target package:", args.output)


if __name__ == "__main__":
    main()
