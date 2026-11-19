#!/usr/bin/env python3
"""Expose YOLOv8 box and class-score branches as separate ONNX outputs."""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def tensor_shapes(graph):
    result = {}
    values = list(graph.input) + list(graph.output) + list(graph.value_info)
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = []
        for dim in tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.dim_value > 0 else None)
        result[value.name] = dims
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--classes", type=int, default=80)
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    model = onnx.shape_inference.infer_shapes(model)
    if len(model.graph.output) != 1:
        raise RuntimeError("Expected the original single YOLOv8 output tensor")
    combined_output = model.graph.output[0]
    combined_name = combined_output.name
    concat = next(
        (node for node in model.graph.node if combined_name in node.output), None
    )
    if concat is None or concat.op_type != "Concat" or len(concat.input) != 2:
        raise RuntimeError("Expected the final YOLO output to be a two-input Concat")

    box_name, score_name = concat.input
    shape_map = tensor_shapes(model.graph)
    combined_shape = shape_map.get(combined_name)
    if not combined_shape or len(combined_shape) != 3:
        raise RuntimeError(f"Could not infer the expected [batch, channels, candidates] output: {combined_name}")
    batch, combined_channels, candidates = combined_shape
    box_shape = shape_map.get(box_name)
    score_shape = shape_map.get(score_name)
    expected_box = [batch, 4, candidates]
    expected_scores = [batch, args.classes, candidates]
    if box_shape and len(box_shape) == 3:
        expected_box = box_shape
    if score_shape and len(score_shape) == 3:
        expected_scores = score_shape
    if combined_channels is not None and combined_channels != 4 + args.classes:
        raise RuntimeError(
            f"Output channels={combined_channels}; expected 4 + classes({args.classes})"
        )

    elem_type = combined_output.type.tensor_type.elem_type or TensorProto.FLOAT
    box_info = helper.make_tensor_value_info(box_name, elem_type, expected_box)
    score_info = helper.make_tensor_value_info(score_name, elem_type, expected_scores)

    # Remove the final concatenation and any duplicate output metadata entries.
    model.graph.node.remove(concat)
    del model.graph.output[:]
    model.graph.output.extend([box_info, score_info])
    output_names = {box_name, score_name}
    kept_value_info = [item for item in model.graph.value_info if item.name not in output_names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)

    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print("Saved validated split-output ONNX:", args.output)
    print("Outputs:", [(box_name, expected_box), (score_name, expected_scores)])


if __name__ == "__main__":
    main()
