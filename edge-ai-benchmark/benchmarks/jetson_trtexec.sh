#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 MODEL.onnx fp32|fp16 [ENGINE.plan]" >&2
  exit 2
fi

model=$1
precision=$2
engine=${3:-"${model%.*}_${precision}.engine"}
trtexec=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}

if [[ ! -x "$trtexec" ]]; then
  echo "trtexec not executable: $trtexec (set TRTEXEC to its path)" >&2
  exit 1
fi
if [[ ! -f "$model" ]]; then
  echo "Model not found: $model" >&2
  exit 1
fi

case "$precision" in
  fp32) precision_flag=() ;;
  fp16) precision_flag=(--fp16) ;;
  *) echo "Precision must be fp32 or fp16" >&2; exit 2 ;;
esac

echo "Building $precision TensorRT engine: $engine"
"$trtexec" \
  --onnx="$model" \
  --saveEngine="$engine" \
  --workspace=1024 \
  "${precision_flag[@]}" \
  --iterations=50 \
  --warmUp=500 \
  --duration=1

echo "Measuring loaded engine: $engine"
"$trtexec" \
  --loadEngine="$engine" \
  --iterations=100 \
  --warmUp=1000 \
  --duration=10
