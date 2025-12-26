import os
import time
import json
import cv2
import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/detect.tflite")
LABELS_PATH = os.getenv("LABELS_PATH", "/app/models/labels.txt")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

SCORE_THRESH = float(os.getenv("SCORE_THRESH", "0.55"))
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "10"))

# Set to arduino_cloud
ALERT_MODE = os.getenv("ALERT_MODE", "arduino_cloud").lower()

# Arduino IoT Cloud provisioning
ARDUINO_DEVICE_ID = os.getenv("ARDUINO_DEVICE_ID", "")
ARDUINO_SECRET_KEY = os.getenv("ARDUINO_SECRET_KEY", "")

# Arduino Cloud variable names (MUST match your Thing)
VAR_PERSON = os.getenv("VAR_PERSON", "person_detected")
VAR_DOG = os.getenv("VAR_DOG", "dog_detected")
VAR_LAST_LABEL = os.getenv("VAR_LAST_LABEL", "last_label")
VAR_LAST_SCORE = os.getenv("VAR_LAST_SCORE", "last_score")
VAR_LAST_TS = os.getenv("VAR_LAST_TS", "last_ts")

# COCO ids (common for SSD models)
COCO_PERSON = 1
COCO_DOG = 18


def load_labels(path: str) -> dict[int, str]:
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            name = line.strip()
            if name:
                labels[i] = name
    return labels


class ArduinoCloudSender:
    def __init__(self):
        if not ARDUINO_DEVICE_ID or not ARDUINO_SECRET_KEY:
            raise RuntimeError("Missing ARDUINO_DEVICE_ID or ARDUINO_SECRET_KEY")

        from arduino_iot_cloud import ArduinoCloudClient

        self.client = ArduinoCloudClient(
            device_id=ARDUINO_DEVICE_ID,
            username=ARDUINO_DEVICE_ID,
            password=ARDUINO_SECRET_KEY,
        )

        # Register variables you will write (must exist in the Thing)
        self.client.register(VAR_PERSON)
        self.client.register(VAR_DOG)
        self.client.register(VAR_LAST_LABEL)
        self.client.register(VAR_LAST_SCORE)
        self.client.register(VAR_LAST_TS)

        # Start background loop
        self.client.start()
        print("Arduino Cloud client started")

    def send(self, payload: dict):
        label = payload.get("label", "")
        score = float(payload.get("score", 0.0))
        ts = int(payload.get("ts", int(time.time())))

        self.client[VAR_PERSON] = (label == "person")
        self.client[VAR_DOG] = (label == "dog")
        self.client[VAR_LAST_LABEL] = label
        self.client[VAR_LAST_SCORE] = score
        self.client[VAR_LAST_TS] = ts

    def send(self, payload: dict):
        """
        payload = {"label": "person"|"dog", "score": float, "ts": int, ...}
        """
        label = payload.get("label", "")
        score = float(payload.get("score", 0.0))
        ts = int(payload.get("ts", int(time.time())))

        # Update your Thing variables
        self.client[VAR_PERSON] = (label == "person")
        self.client[VAR_DOG] = (label == "dog")
        self.client[VAR_LAST_LABEL] = label
        self.client[VAR_LAST_SCORE] = score
        self.client[VAR_LAST_TS] = ts


def main():
    # TFLite runtime
    from tflite_runtime.interpreter import Interpreter

    _ = load_labels(LABELS_PATH)  # not used, but kept since you had it

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    # Typical SSD input: [1, 300, 300, 3] uint8
    in_h = in_details[0]["shape"][1]
    in_w = in_details[0]["shape"][2]
    in_dtype = in_details[0]["dtype"]

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check /dev/video0 mapping and permissions.")

    sender = None
    if ALERT_MODE == "arduino_cloud":
        sender = ArduinoCloudSender()
    else:
        raise RuntimeError("Set ALERT_MODE=arduino_cloud (this version only sends to Arduino Cloud).")

    last_alert_ts = 0

    print(f"Running. Camera={CAMERA_INDEX} model={MODEL_PATH} input=({in_w}x{in_h}) dtype={in_dtype}")
    print("Arduino Cloud sender active. Writing variables:",
          VAR_PERSON, VAR_DOG, VAR_LAST_LABEL, VAR_LAST_SCORE, VAR_LAST_TS)

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue

        # Resize to model input
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (in_w, in_h))

        input_tensor = np.expand_dims(resized, axis=0)
        if in_dtype == np.float32:
            input_tensor = (input_tensor / 255.0).astype(np.float32)
        else:
            input_tensor = input_tensor.astype(in_dtype)

        interpreter.set_tensor(in_details[0]["index"], input_tensor)
        interpreter.invoke()

        # Typical SSD outputs:
        # boxes: [1, N, 4], classes: [1, N], scores: [1, N], count: [1]
        boxes = interpreter.get_tensor(out_details[0]["index"])[0]
        classes = interpreter.get_tensor(out_details[1]["index"])[0].astype(np.int32)
        scores = interpreter.get_tensor(out_details[2]["index"])[0]

        # Find best person/dog detection above threshold
        best = None
        for cls, score, box in zip(classes, scores, boxes):
            if score < SCORE_THRESH:
                continue
            if cls in (COCO_PERSON, COCO_DOG):
                if best is None or score > best["score"]:
                    best = {"class_id": int(cls), "score": float(score), "box": box.tolist()}

        now = time.time()
        if best and (now - last_alert_ts) >= COOLDOWN_SEC:
            label = "person" if best["class_id"] == COCO_PERSON else "dog"
            payload = {
                "event": "object_detected",
                "label": label,
                "score": best["score"],
                "ts": int(now),
            }
            print("ALERT:", payload)
            try:
                sender.send(payload)
                last_alert_ts = now
            except Exception as e:
                print("Arduino Cloud update failed:", e)

        time.sleep(0.05)


if __name__ == "__main__":
    main()
