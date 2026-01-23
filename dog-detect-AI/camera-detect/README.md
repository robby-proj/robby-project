# :dog: Camera Dog & Person Detection (Arduino UNO Q)
This project runs a real-time object detection pipeline on an Arduino UNO Q using TensorFlow Lite, OpenCV, and Docker.
It detects dogs and people from a USB camera and publishes detection states to Arduino IoT Cloud.
⸻
:sparkles: Features
1. :brain: TensorFlow Lite object detection (COCO model)
2. :movie_camera: USB camera input (OpenCV)
3. :dog2: Detects dog (COCO class 17)
4. :standing_person: Detects person (COCO class 0)
5. :cloud: Sends detection results to Arduino IoT Cloud
6. :whale: Fully containerized with Docker
7. :rocket: Auto-start ready via Docker Compose

```
camera-detect/
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── models/
│   ├── detect.tflite
│   └── labelmap.txt  
└── README.md
```

# :gear: Requirements

Hardware
1. Arduino UNO Q
2. USB camera (UVC compatible)
3. Internet connection (for Arduino IoT Cloud)
4. USB-C hub with power supply

 Software
1. Debian (UNO Q default)
2. Docker
3. Docker Compose

⸻

# :cloud: Arduino IoT Cloud Setup

Create a THING in Arduino IoT Cloud with the following variables:

* Variable: person_detected , type : Boolean , Permission: Read & Write
*	Variable: dog_detected , type : Boolean , Permission: Read & Write
*	Variable: dog_detected , type : Boolean , Permission: Read & Write
*	Variable: last_label , type : String , Permission: Read & Write
*	Variable: last_score , type : Float , Permission: Read & Write
*	Variable: last_ts , type : Integer , Permission: Read & Write

Copy:
	* Device ID
	* Secret Key
You will use them in docker-compose.yml

# :whale: Docker Setup

On the docker-compose.yml configure your ARDUINO_DEVICE_ID , ARDUINO_SECRET_KEY 

```
version: "3.8"

services:
  camdetect:
    build: .
    container_name: camdetect
    restart: unless-stopped
    devices:
      - /dev/video0:/dev/video0
    environment:
      ARDUINO_DEVICE_ID: "YOUR_DEVICE_ID"
      ARDUINO_SECRET_KEY: "YOUR_SECRET_KEY"

      VAR_PERSON: "person_detected"
      VAR_DOG: "dog_detected"
      VAR_LAST_LABEL: "last_label"
      VAR_LAST_SCORE: "last_score"
      VAR_LAST_TS: "last_ts"
```

# :brain: Model Details
* Model: TensorFlow Lite SSD
* Input size: 300x300
* COCO class IDs used:
   - 0 → person
   - 17 → dog
 
# :broccoli: Downloading Models

Run these commands in the directory:
```
cd ~/camera-detect
mkdir -p models
cd models
```
Model
```
wget -O detect.tflite https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
```
Extract the zip file
```
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
```
The model file inside is usually "detect.tflite" already
```
ls -la
```
Get the labels
```
wget -O labels.txt https://storage.googleapis.com/download.tensorflow.org/data/coco_labels.txt
```
Ensure models and labels are exactly here.
```
# ./models/detect.tflite
# ./models/labels.txt

ls -la ./models
```
# :arrow_forward: Running the Project

Build & Start
```
docker compose up -d --build
```
view Logs
```
docker logs -f camdetect
```
You should see logs like :
```
DETECTION: 17 0.58
INFO: DETECTED: dog score=0.580
```

# :stopwatch: 6-Hour Detection Report

This project includes an automatic 6-hour activity report generated directly by the detection container.

:page_facing_up: Report Format

Every 6 hours, the system sends a summary report to Arduino IoT Cloud in the following format:

```
report ( person detected: 1.2 hours, dog detected: 4.6 hours )
```

# :brain: How the Report Works
- The camera runs continuous object detection.
- Each frame contributes time to:
	- Person detected
	- Dog detected
 	- No detection
- Time is accumulated internally.
- Every 6 hours:
	-  Totals are calculated
 	-  The report string is sent to the cloud
  	-  Counters reset for the next period

This allows you to:
* Measure how long your dog stays in bed
* Detect human activity vs pet activity
* Track behavior patterns over time
