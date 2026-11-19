# Benchmark Procedure and Reproduction Notes

## 1. Objective

Evaluate a practical conversion and inference workflow for YOLOv8n across:

1. Arduino Ventuno Q using Qualcomm AI Hub / QAIRT and ONNX Runtime QNN Execution Provider (QNN EP).
2. NVIDIA Jetson Nano using TensorRT.
3. Arduino UNO Q using ONNX Runtime CPU and NCNN CPU/Vulkan paths.

The primary comparison is **precision change within each platform/runtime**, because the tested input tensors, timing boundaries, and runtime APIs were not aligned well enough for a strict cross-board ranking.

## 2. Model and workload

- Model family: Ultralytics YOLOv8n object detector, exported as ONNX.
- Input tensor: `images`, shape `[1, 3, 640, 640]`, FP32 at the interface.
- Output before splitting: `[1, 84, 8400]` (4 box values and 80 class scores per candidate).
- Batch: 1.
- Dataset used for Ventuno calibration: 32 images selected from COCO128 at a 4-image stride.
- Pilot ground-truth evaluation: 31 labeled images; one selected image had no label file and was skipped.

## 3. Platform setup

| Platform | Hardware / OS | Runtime path |
|---|---|---|
| Ventuno Q | Qualcomm QCS8275 / IQ8, Ubuntu 24.04, 16 GB class | QAIRT 2.49; ONNX Runtime 1.23 with QNN EP |
| Jetson Nano | NVIDIA Tegra X1 Maxwell, 4 GB, JetPack 4 / L4T R32.7.4 | TensorRT 8.2.1.8, `trtexec` |
| UNO Q | QRB2210 family, 4 GB class, Debian 13, Adreno 702 | ONNX Runtime CPU EP; NCNN CPU and Vulkan |

## 4. Ventuno Q conversion, quantization, and execution

### 4.1 FP32 baseline

1. Start with YOLOv8n ONNX exported from the PyTorch model.
2. Validate the graph with ONNX checker and create an AI Hub-ready copy. The first compile identified duplicate output metadata: `output0` appeared both in graph outputs and `value_info`. Removing the duplicate metadata fixed the compile issue without changing model computations.
3. Compile for the exact AI Hub target `Arduino VENTUNO Q`, OS `24.04`, targeting ONNX/QNN.
4. Run on-board using ONNX Runtime providers `QNNExecutionProvider` and `CPUExecutionProvider`.
5. Enable ONNX Runtime profiling and check node-level provider records. The FP32 result reported QNN node execution; a provider being listed by ORT alone is not sufficient proof that all graph nodes ran on it.

### 4.2 W8A8 quantization

1. Use representative COCO128 images as calibration data, preprocessed to the model input shape `[1, 3, 640, 640]` and float32 interface values.
2. Submit AI Hub quantization with INT8 weights and INT8 activations (W8A8), then compile the quantized model for the Ventuno Q.
3. The initial one-output W8A8 model returned zero class scores. Investigation found a single dequantization scale of approximately `2.509049` applied to the combined output tensor. That tensor mixed box coordinates around 0–640 with class scores in 0–1, so the scale was too coarse for class scores.
4. Split the graph output into separate box and score branches before quantization. That lets the branches use appropriate quantization scales. Reconstruct the original `[1,84,8400]` output for comparison or postprocessing.
5. Confirm that the score branch is nonzero and that the QNN EP appears in the runtime profile. Compare detections and labeled metrics to the FP32 baseline.

### 4.3 Timing method recorded

The reported Ventuno inference loop used five warm-up calls followed by 30 timed inference calls on zero-valued input tensors. It reported mean, median, and p95 from host-side wall-clock timing around `session.run`. Model compilation/initialization is excluded from steady-state latency; image capture and detection postprocessing are also excluded.

## 5. Jetson Nano TensorRT procedure

The board was identified as Jetson Nano Developer Kit, Tegra X1, JetPack/L4T R32.7.4, TensorRT 8.2.1.8. The ONNX model was transferred to the Jetson. TensorRT engine build and loaded-engine inference were measured separately.

Example engine-build commands (paths are illustrative):

```bash
# FP32
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/hiwonder/yolov8n.onnx \
  --saveEngine=/home/hiwonder/yolov8n_fp32.engine \
  --workspace=1024 --iterations=50 --warmUp=500 --duration=1

# FP16-enabled TensorRT build
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/hiwonder/yolov8n.onnx \
  --saveEngine=/home/hiwonder/yolov8n_fp16.engine \
  --workspace=1024 --fp16 --iterations=50 --warmUp=500 --duration=1
```

Loaded-engine performance was then sampled using `--loadEngine`, `--iterations=100`, `--warmUp=1000`, and `--duration=10`. `trtexec` generated random input values. The FP16 label means FP16 was enabled; unsupported operations may still run in FP32. Reported end-to-end latency includes host/runtime and transfer overhead. The loaded-engine run is the latency reported in the result tables; engine build time is not inference latency.

## 6. UNO Q NCNN and ONNX Runtime procedure

1. Run ONNX Runtime 1.23 CPU with the model input `[1,3,640,640]`. The installed provider list was `AzureExecutionProvider` and `CPUExecutionProvider`; no Qualcomm QNN provider was installed on this UNO Q setup.
2. Convert the ONNX model to NCNN `.param` and `.bin` files with PNNX. The converted graph reported 8.855 GFLOP for the 640 × 640 input.
3. Measure NCNN CPU inference with four threads. For INT8, build a calibration table using representative images (KL-divergence method) and quantize the NCNN model with the NCNN tools `ncnn2table` and `ncnn2int8`.
4. For the Vulkan experiment, select NCNN Vulkan device 0 (Turnip Adreno 702), enable NCNN FP16 options, and measure separately from CPU.
5. The UNO Q latency loops used zero-valued input tensors. The Vulkan result is exploratory; device selection does not itself establish that every graph operator ran on Adreno. No layer-placement trace was captured.

The recorded INT8 weight file was 3.2 MB versus 13.0 MB FP32 (about 75% smaller). The corresponding UNO Q INT8 detection quality has not yet been validated against labeled ground truth.

## 7. Quality evaluation

- Ventuno calibration set and evaluation set are separate roles: calibration data configures quantization scales; labeled images measure task behavior.
- The labeled pilot evaluated 31 images with available YOLO-format labels (216 ground-truth objects total).
- Reported precision, recall, and F1:
  - QNN FP32: 84.73%, 51.39%, 63.98%.
  - QNN W8A8: 83.33%, 48.61%, 61.40%.
- Separately, W8A8 matched FP32 at 89.3% on a 32-image comparison using same-class matches at IoU ≥ 0.5. This measures agreement with FP32, not correctness against ground truth.
- The original one-output W8A8 attempt was rejected because class scores collapsed to zero. Only the split-output corrected model is used for the quality and final latency claims.

For a fully repeatable quality rerun, preserve the exact confidence threshold, NMS IoU threshold, class handling, image preprocessing, and metric implementation alongside the code. Those complete evaluation-script settings are not available in the captured run record.

## 8. Measurement cautions

- Ventuno and UNO Q ORT/NCNN loops used zero tensors; Jetson `trtexec` used random input. These runs measured model execution rather than camera-to-result application latency.
- The boards used different runtimes, timing APIs, input-value generation, and measurement windows. Use same-board comparisons to discuss precision changes. Treat cross-board chart ordering as directional.
- We did not record matched power/thermal conditions, sustained temperature, or energy per inference.
- The concurrent Ventuno NPU vision + CPU/GPU LLM test has not been run. Do not claim that a concurrent workload leaves vision performance unchanged until it is measured.
- Peak TOPS/GFLOPS were not measured in these experiments. The PNNX conversion estimate of 8.855 GFLOP describes this model/input estimate, not measured board throughput.

## 9. Recommended controlled rerun

For a stronger sales demonstration, use the same saved input images and preprocessing on all devices; match batch, resolution, confidence/NMS settings, and measurement boundary; warm up at least 50 calls; measure at least 200 calls; report median, p95, throughput, and full camera-to-result latency separately. Record device power mode, ambient temperature, board temperature, and power draw. Then run the Ventuno vision model alone and concurrently with llama.cpp, reporting vision latency/throughput and LLM tokens per second before and during concurrency.

## 10. Cleanup and secret handling

Keep downloaded model files, datasets, compiled engines, NCNN weight binaries, AI Hub credentials, and runtime profile dumps out of Git. The repository `.gitignore` lists common sensitive or bulky artifacts. Revoke/rotate any credential accidentally committed.

## 11. Runnable scripts in this repository

The companion code is under [`../benchmarks/`](../benchmarks/) and [`../aihub/`](../aihub/). It is parameterized so model paths, AI Hub model IDs, image folders, iteration counts, and runtime choices are supplied at run time instead of being embedded in the source. Install host-side dependencies with `python -m pip install -r aihub/requirements.txt`. On UNO Q, install `python -m pip install -r benchmarks/requirements-ncnn.txt`. For ONNX Runtime, use the board-specific runtime build that exposes the provider being measured; the generic PyPI ONNX Runtime package does not provide QNN EP.

To recreate the ONNX input from an Ultralytics PyTorch checkpoint, install `aihub/requirements-export.txt` in a separate export environment, then run:

```bash
python aihub/export_yolov8_onnx.py --weights /path/to/yolov8n.pt --output-dir /path/to/exported
```

Use the same checkpoint, Ultralytics version, input size, and export options when comparing to a customer's baseline. Exporting a checkpoint with a different library version can change the graph.

### Ventuno Q: calibrate W8A8 and compile

On the host with `qai-hub` installed and configured, first compile the split-output FP32 ONNX model. For example:

```bash
python aihub/split_yolov8_outputs.py \
  --input /path/to/yolov8n.onnx \
  --output /path/to/yolov8n_split_outputs.onnx

python aihub/compile_ventuno.py \
  --onnx /path/to/yolov8n_split_outputs.onnx \
  --output /path/to/ventuno_fp32_package
```

The split script replaces YOLOv8's terminal combined output with separate box and class-score outputs. The compile script prints the AI Hub source model ID. Use that ID to quantize with representative images. Quantizing the original combined output reproduced the zero-score failure. AI Hub appends the target package extension to the output name; inspect the downloaded archive with `unzip -l` before extracting it.

```bash
python aihub/quantize_w8a8.py \
  --model-id MODEL_ID_FROM_COMPILE \
  --images-dir /path/to/coco128/images/train2017 \
  --name VentunoQ-YOLOv8n-W8A8
```

The quantizer prints the resulting model ID. Compile that model for the Ventuno Q:

```bash
python aihub/compile_ventuno.py \
  --model-id QUANTIZED_MODEL_ID \
  --output /path/to/ventuno_w8a8_package
```

Copy the resulting ONNX package and its companion `model.data` file to the Ventuno Q, then use the on-device runner:

```bash
python benchmarks/ort_benchmark.py \
  --model /path/to/model.onnx \
  --providers QNNExecutionProvider CPUExecutionProvider \
  --warmup 5 --iterations 30 --profile
```

For the CPU baseline, use `--providers CPUExecutionProvider`. The ORT script uses zeros unless an `.npy` input tensor is supplied with `--input-npy`. Its timings cover `session.run`, not image preprocessing or detection postprocessing.

### Jetson Nano: TensorRT build and loaded-engine timing

Copy the ONNX model to the Nano, then run:

```bash
bash benchmarks/jetson_trtexec.sh /path/to/yolov8n.onnx fp32
bash benchmarks/jetson_trtexec.sh /path/to/yolov8n.onnx fp16
```

The script builds an engine and times the loaded engine. Set `TRTEXEC` if `trtexec` is installed outside `/usr/src/tensorrt/bin/trtexec`.

### UNO Q: NCNN timing

Convert the model to NCNN `.param` and `.bin` artifacts. PNNX stores FP16 weights by default; specify `fp16=0` for the FP32 reference used here:

```bash
pnnx /path/to/yolov8n.onnx inputshape=[1,3,640,640] fp16=0
```

To calibrate and generate the INT8 NCNN model, create a sorted image list (one image path per line) and run the NCNN tools:

```bash
find /path/to/coco128/images/train2017 -type f -name '*.jpg' | sort > ncnn_calibration_images.txt

ncnn2table yolov8n.ncnn.param yolov8n.ncnn.bin \
  ncnn_calibration_images.txt yolov8n_int8.table \
  mean=[0,0,0] norm=[0.0039215686,0.0039215686,0.0039215686] \
  shape=[640,640,3] pixel=RGB thread=4 method=kl

ncnn2int8 yolov8n.ncnn.param yolov8n.ncnn.bin \
  yolov8n_int8.ncnn.param yolov8n_int8.ncnn.bin yolov8n_int8.table
```

If the tools are not in `PATH`, set `NCNN_TOOLS_DIR` and use the helper script `benchmarks/ncnn_int8_convert.sh`. Run the converted models on UNO Q:

```bash
python benchmarks/ncnn_benchmark.py \
  --param yolov8n_fp32.ncnn.param --weights yolov8n_fp32.ncnn.bin \
  --precision fp32 --device cpu --threads 4

python benchmarks/ncnn_benchmark.py \
  --param yolov8n_int8.ncnn.param --weights yolov8n_int8.ncnn.bin \
  --precision int8 --device cpu --threads 4
```

For the experimental Adreno path, select `--device vulkan --gpu-id 0 --precision fp16`. The NCNN runner reports the selected Vulkan device but does not claim that every layer executes on the GPU.

### Re-running quality metrics

The run history did not preserve every confidence/NMS setting from the original quality-evaluation script. The reported pilot metrics are retained as results, but reproducing them exactly requires recording those settings and the metric code. Do not use the latency scripts as an accuracy evaluator. These scripts document and automate conversion, calibration, compilation, and latency measurement; verify each script on the target hardware before relying on a rerun result.
