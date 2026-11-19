#!/usr/bin/env python3
"""Measure ONNX Runtime inference latency and, optionally, node providers."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_shape(value):
    return tuple(int(part.strip()) for part in value.split(","))


def tensor_dtype(type_name):
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
    }
    if type_name not in mapping:
        raise ValueError(f"Unsupported input tensor type: {type_name}")
    return mapping[type_name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
    parser.add_argument("--input-name")
    parser.add_argument("--shape", help="Override input shape, e.g. 1,3,640,640")
    parser.add_argument("--input-npy", type=Path, help="Optional saved input tensor")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be >= 0 and --iterations must be >= 1")

    options = ort.SessionOptions()
    options.enable_profiling = args.profile
    session = ort.InferenceSession(
        str(args.model), sess_options=options, providers=args.providers
    )
    model_input = next(
        (item for item in session.get_inputs() if item.name == args.input_name),
        None,
    ) if args.input_name else session.get_inputs()[0]
    if model_input is None:
        raise ValueError(f"Input name not found: {args.input_name}")

    shape = parse_shape(args.shape) if args.shape else tuple(
        dim if isinstance(dim, int) and dim > 0 else None for dim in model_input.shape
    )
    if any(dim is None for dim in shape):
        raise ValueError("Model has dynamic input dimensions; pass --shape explicitly")
    dtype = tensor_dtype(model_input.type)
    if args.input_npy:
        tensor = np.load(args.input_npy)
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"Input tensor shape {tensor.shape} does not match {shape}")
        tensor = tensor.astype(dtype, copy=False)
    else:
        tensor = np.zeros(shape, dtype=dtype)

    feed = {model_input.name: tensor}
    for _ in range(args.warmup):
        outputs = session.run(None, feed)

    elapsed_ms = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        outputs = session.run(None, feed)
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    print(f"Input: {model_input.name} {tuple(tensor.shape)} {tensor.dtype}")
    print("Outputs:", [tuple(output.shape) for output in outputs])
    print("Session providers:", session.get_providers())
    print(
        "Latency ms: "
        f"mean={np.mean(elapsed_ms):.2f}, "
        f"median={np.median(elapsed_ms):.2f}, "
        f"p95={np.percentile(elapsed_ms, 95):.2f}"
    )

    if args.profile:
        profile_path = Path(session.end_profiling())
        events = json.loads(profile_path.read_text())
        summary = {}
        for event in events:
            if event.get("cat") != "Node" or event.get("ph") != "X":
                continue
            provider = event.get("args", {}).get("provider", "unknown")
            stats = summary.setdefault(provider, {"nodes": 0, "duration_ms": 0.0})
            stats["nodes"] += 1
            stats["duration_ms"] += event.get("dur", 0.0) / 1000.0
        print("Node provider summary:", json.dumps(summary, sort_keys=True))
        print("Profile:", profile_path)


if __name__ == "__main__":
    main()
