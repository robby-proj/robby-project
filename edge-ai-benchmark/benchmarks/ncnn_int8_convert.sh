#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 MODEL.param MODEL.bin IMAGE_LIST OUTPUT_PREFIX" >&2
  exit 2
fi

param=$1
weights=$2
image_list=$3
prefix=$4
tools_dir=${NCNN_TOOLS_DIR:-}
ncnn2table=${NCNN2TABLE:-${tools_dir:+$tools_dir/}ncnn2table}
ncnn2int8=${NCNN2INT8:-${tools_dir:+$tools_dir/}ncnn2int8}
table="${prefix}.table"
out_param="${prefix}.param"
out_weights="${prefix}.bin"

for file in "$param" "$weights" "$image_list"; do
  [[ -f "$file" ]] || { echo "Required file not found: $file" >&2; exit 1; }
done

"$ncnn2table" \
  "$param" "$weights" "$image_list" "$table" \
  mean=[0,0,0] \
  norm=[0.0039215686,0.0039215686,0.0039215686] \
  shape=[640,640,3] pixel=RGB thread=4 method=kl

"$ncnn2int8" "$param" "$weights" "$out_param" "$out_weights" "$table"
echo "Wrote INT8 NCNN model: $out_param and $out_weights"
