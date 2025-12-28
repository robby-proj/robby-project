mkdir -p camera-detect/models
cd camera-detect/models

# Download a common SSD MobileNet V1 model + labels
wget -O detect.tflite https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip detect.tflite -d tmp || true
# The zip usually contains a .tflite; if unzip didn't work because it's not a zip on your side, use the alternative below.

# Alternative (most reliable): use a known raw tflite + labels hosted on GitHub
cd ..
