
# Exporting and Organizing Models on the UNO Q

Your recommended current directory layout:

```text
/home/arduino
├── ArduinoApps
├── models
├── yolo-export-venv
└── yzma
```

Your app folder is:

```text
/home/arduino/ArduinoApps/unoq-vision-app
```

Your model storage folder is:

```text
/home/arduino/models
```

Your `yzma` repo is:

```text
/home/arduino/yzma
```

---

## 1. Where each model should live

### YOLO detector files
Store these in:

```text
/home/arduino/models/coco
```

Expected files:

```text
/home/arduino/models/coco/yolov8n.onnx
/home/arduino/models/coco/coco.names
```

### VLM files
Store these in:

```text
/home/arduino/models
```

Expected files:

```text
/home/arduino/models/SmolVLM2-500M-Video-Instruct-Q8_0.gguf
/home/arduino/models/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf
```

### yzma runtime
Keep `yzma` here:

```text
/home/arduino/yzma
```

Make sure the library folder exists:

```text
/home/arduino/yzma/lib
```

You can verify it with:

```bash
cd /home/arduino/yzma
ls
```

---

## 2. Verify your current model layout

Run:

```bash
cd /home/arduino/models
ls
```

To inspect the COCO detector folder:

```bash
ls /home/arduino/models/coco
```

---

## 3. How to export the YOLO ONNX model

Because exporting directly on the UNO Q was unreliable, the working method is:

1. export `yolov8n.onnx` on your Mac or another stronger machine
2. copy it into the UNO Q model folder

### On your Mac
Create and activate a Python environment:

```bash
python3 -m venv ~/yolo-export-venv
source ~/yolo-export-venv/bin/activate
pip install --upgrade pip
pip install ultralytics onnx
mkdir -p ~/yolo-export
cd ~/yolo-export
```

Export with Python:

```bash
python3 - <<'PY'
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640)
PY
```

Verify the file exists:

```bash
ls -lh ~/yolo-export
find ~/yolo-export -name "*.onnx"
```

You want an output file named:

```text
yolov8n.onnx
```

---

## 4. Copy the exported ONNX file to the UNO Q

From your Mac:

```bash
scp ~/yolo-export/yolov8n.onnx arduino@<UNOQ_IP>:/home/arduino/models/coco/
```

Example:

```bash
scp ~/yolo-export/yolov8n.onnx arduino@192.168.12.236:/home/arduino/models/coco/
```

Then on the UNO Q verify:

```bash
ls -lh /home/arduino/models/coco/yolov8n.onnx
```

---

## 5. Create `coco.names` on the UNO Q

If needed:

```bash
mkdir -p /home/arduino/models/coco
cat > /home/arduino/models/coco/coco.names <<'EOF'
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
EOF
```

Verify:

```bash
wc -l /home/arduino/models/coco/coco.names
head -5 /home/arduino/models/coco/coco.names
```

---

## 6. Verify the VLM model files

On the UNO Q:

```bash
ls -lh /home/arduino/models
```

You should see:

```text
SmolVLM2-500M-Video-Instruct-Q8_0.gguf
mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf
```

---

## 7. Verify the `yzma` folder and libraries

Run:

```bash
cd /home/arduino/yzma
ls
ls /home/arduino/yzma/lib
```

If the `lib` folder is missing or incomplete:

```bash
export YZMA_LIB=/home/arduino/yzma/lib
yzma install -u --processor cpu --os trixie
```

---

## 8. How your app consumes the models

Your app folder is:

```text
/home/arduino/ArduinoApps/unoq-vision-app
```

Your `docker-compose.yml` should point to the model locations like this:

```yaml
YOLO_ONNX_PATH: /models/coco/yolov8n.onnx
COCO_NAMES_PATH: /models/coco/coco.names

VLM_MODEL_PATH: /models/SmolVLM2-500M-Video-Instruct-Q8_0.gguf
VLM_MMPROJ_PATH: /models/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf
YZMA_DIR: /yzma
```

Because the compose file mounts:

```yaml
- /home/arduino/models:/models
- /home/arduino/yzma:/yzma
```

the container sees the host folders as `/models` and `/yzma`.

---

## 9. Quick verification checklist

On the UNO Q host:

```bash
ls -lh /home/arduino/models/coco/yolov8n.onnx
ls -lh /home/arduino/models/coco/coco.names
ls -lh /home/arduino/models/SmolVLM2-500M-Video-Instruct-Q8_0.gguf
ls -lh /home/arduino/models/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf
ls -lh /home/arduino/yzma/lib
```

Inside the running container:

```bash
docker exec -it unoq_vision_vlm sh -c '
ls -lh /models/coco
ls -lh /models
ls -lh /yzma/lib
'
```

---

## 10. Useful paths summary

```text
YOLO ONNX:
  /home/arduino/models/coco/yolov8n.onnx

COCO labels:
  /home/arduino/models/coco/coco.names

VLM model:
  /home/arduino/models/SmolVLM2-500M-Video-Instruct-Q8_0.gguf

VLM mmproj:
  /home/arduino/models/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf

yzma repo:
  /home/arduino/yzma

yzma lib:
  /home/arduino/yzma/lib

app folder:
  /home/arduino/ArduinoApps/unoq-vision-app
```
