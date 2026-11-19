#!/usr/bin/env python3
"""Benchmark a converted NCNN model on CPU or a selected Vulkan device."""

import argparse
import time

import numpy as np
import ncnn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--param", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", choices=("cpu", "vulkan"), default="cpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp32")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be >= 0 and --iterations must be >= 1")

    gpu_created = False
    net = None
    if args.device == "vulkan":
        ncnn.create_gpu_instance()
        gpu_created = True
        gpu_count = ncnn.get_gpu_count()
        if args.gpu_id < 0 or args.gpu_id >= gpu_count:
            ncnn.destroy_gpu_instance()
            raise ValueError(f"GPU index {args.gpu_id} invalid; visible GPU count={gpu_count}")

    try:
        net = ncnn.Net()
        net.opt.num_threads = args.threads
        if args.device == "vulkan":
            net.opt.use_vulkan_compute = True
            net.set_vulkan_device(args.gpu_id)
        if args.precision == "fp16":
            net.opt.use_fp16_packed = True
            net.opt.use_fp16_storage = True
            net.opt.use_fp16_arithmetic = True
        elif args.precision == "int8":
            net.opt.use_int8_inference = True
            net.opt.use_int8_storage = True
            net.opt.use_int8_packed = True
            net.opt.use_int8_arithmetic = True

        if net.load_param(args.param) != 0:
            raise RuntimeError(f"Could not load NCNN parameter file: {args.param}")
        if net.load_model(args.weights) != 0:
            raise RuntimeError(f"Could not load NCNN weights: {args.weights}")
        input_names = net.input_names()
        output_names = net.output_names()
        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError(
                f"Expected one model input/output, got {input_names} / {output_names}"
            )

        input_mat = ncnn.Mat(args.width, args.height, args.channels)
        input_mat.fill(0.0)

        def infer():
            extractor = net.create_extractor()
            if extractor.input(input_names[0], input_mat) != 0:
                raise RuntimeError("NCNN input failed")
            code, output = extractor.extract(output_names[0])
            if code != 0:
                raise RuntimeError("NCNN inference failed")
            return output

        for _ in range(args.warmup):
            output = infer()
        timings = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            output = infer()
            timings.append((time.perf_counter() - start) * 1000.0)

        print(f"Device: {args.device}; precision setting: {args.precision}")
        if args.device == "vulkan":
            print(
                f"Selected Vulkan device: {ncnn.get_gpu_info(args.gpu_id).device_name()} "
                f"(index {args.gpu_id})"
            )
        print(f"NCNN output: dims={output.dims}, w={output.w}, h={output.h}, c={output.c}")
        print(
            "Latency ms: "
            f"mean={np.mean(timings):.2f}, "
            f"median={np.median(timings):.2f}, "
            f"p95={np.percentile(timings, 95):.2f}"
        )
    finally:
        if args.device == "vulkan" and gpu_created:
            if net is not None:
                del net
            ncnn.destroy_gpu_instance()


if __name__ == "__main__":
    main()
