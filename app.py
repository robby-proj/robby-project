import os
import time
import threading
import logging
import cv2
import numpy as np

from arduino_iot_cloud import ArduinoCloudClient


# -----------------------------
# Config
# -----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/detect.tflite")
LABELS_PATH = os.getenv("LABELS_PATH", "/app/models/labels.txt")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

SCORE_THRESH = float(os.getenv("SCORE_THRESH", "0.55"))
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "10"))

# Report timing
REPORT_EVERY_SEC = int(os.getenv("REPORT_EVERY_SEC", str(6 * 60 * 60)))  # 6 hours default

# COCO IDs
COCO_PERSON = 0
COCO_DOG = 17

# Arduino IoT Cloud credentials (from docker-compose.yml env vars)
ARDUINO_DEVICE_ID = os.getenv("ARDUINO_DEVICE_ID", "")
ARDUINO_SECRET_KEY = os.getenv("ARDUINO_SECRET_KEY", "")

# Arduino Cloud variable names (must match your Thing variables exactly)
VAR_PERSON = os.getenv("VAR_PERSON", "person_detected")
VAR_DOG = os.getenv("VAR_DOG", "dog_detected")
VAR_LAST_LABEL = os.getenv("VAR_LAST_LABEL", "last_label")
VAR_LAST_SCORE = os.getenv("VAR_LAST_SCORE", "last_score")
VAR_LAST_TS = os.getenv("VAR_LAST_TS", "last_ts")
VAR_HEARTBEAT = os.getenv("VAR_HEARTBEAT", "")  # optional
VAR_REPORT = os.getenv("VAR_REPORT", "report")  # <-- create a String variable named "report" in Arduino Cloud


# -----------------------------
# Helpers
# -----------------------------
def load_labels(path: str) -> dict[int, str]:
    labels = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                name = line.strip()
                if name:
                    labels[i] = name
    except FileNotFoundError:
        logging.warning("labels file not found at %s (continuing without it)", path)
    return labels


class ArduinoCloudSender:
    """
    Arduino IoT Cloud client using sync_mode=True + update loop.
    """
    def __init__(self):
        if not ARDUINO_DEVICE_ID or not ARDUINO_SECRET_KEY:
            raise RuntimeError("Missing ARDUINO_DEVICE_ID / ARDUINO_SECRET_KEY env vars")

        logging.basicConfig(
            datefmt="%H:%M:%S",
            format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
            level=logging.INFO,
        )

        self.client = ArduinoCloudClient(
            device_id=ARDUINO_DEVICE_ID,
            username=ARDUINO_DEVICE_ID,
            password=ARDUINO_SECRET_KEY,
            sync_mode=True,
        )

        # Register variables (must exist in the Thing)
        self.client.register(VAR_PERSON)
        self.client.register(VAR_DOG)
        self.client.register(VAR_LAST_LABEL)
        self.client.register(VAR_LAST_SCORE)
        self.client.register(VAR_LAST_TS)

        if VAR_HEARTBEAT:
            self.client.register(VAR_HEARTBEAT)

        # report string
        if VAR_REPORT:
            self.client.register(VAR_REPORT)

        self.client.start()

        self._stop = False
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

        logging.info("Arduino Cloud sender started (sync_mode=True)")

    def _update_loop(self):
        while not self._stop:
            try:
                self.client.update()
            except Exception as e:
                logging.error("Arduino Cloud update() failed: %r", e)
            time.sleep(0.1)

    def close(self):
        self._stop = True

    def heartbeat(self):
        if VAR_HEARTBEAT:
            self.client[VAR_HEARTBEAT] = int(time.time())

    def send_detection(self, label: str, score: float, ts: int):
        self.client[VAR_PERSON] = (label == "person")
        self.client[VAR_DOG] = (label == "dog")
        self.client[VAR_LAST_LABEL] = str(label)
        self.client[VAR_LAST_SCORE] = float(score)
        self.client[VAR_LAST_TS] = int(ts)

    def send_report(self, report_text: str, ts: int):
        if VAR_REPORT:
            self.client[VAR_REPORT] = report_text
        # Also update last_ts so you can see when it reported
        self.client[VAR_LAST_TS] = int(ts)


def fmt_hours(seconds: float) -> str:
    return f"{seconds / 3600.0:.1f}"


def main():
    from tflite_runtime.interpreter import Interpreter

    logging.basicConfig(level=logging.INFO)
    _ = load_labels(LABELS_PATH)

    sender = ArduinoCloudSender()

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    in_h = int(in_details[0]["shape"][1])
    in_w = int(in_details[0]["shape"][2])
    in_dtype = in_details[0]["dtype"]

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check /dev/video0 mapping and permissions.")

    logging.info("Running. Camera=%s model=%s input=(%sx%s) dtype=%s",
                 CAMERA_INDEX, MODEL_PATH, in_w, in_h, in_dtype)
    logging.info("Arduino Cloud vars: %s %s %s %s %s report=%s",
                 VAR_PERSON, VAR_DOG, VAR_LAST_LABEL, VAR_LAST_SCORE, VAR_LAST_TS, VAR_REPORT)

    last_publish_ts = 0
    last_hb = 0

    # For time accounting
    last_loop_time = time.time()
    dog_seconds = 0.0
    person_seconds = 0.0

    # Report scheduler
    next_report_time = time.time() + REPORT_EVERY_SEC

    # For "No detection" publishing (rate-limited)
    last_no_det_publish = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue

        now = time.time()
        dt = max(0.0, now - last_loop_time)
        last_loop_time = now

        # heartbeat every 10 seconds (optional)
        if VAR_HEARTBEAT and (now - last_hb) > 10:
            try:
                sender.heartbeat()
                last_hb = now
            except Exception as e:
                logging.error("Heartbeat failed: %r", e)

        # Preprocess
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (in_w, in_h))

        input_tensor = np.expand_dims(resized, axis=0)
        if in_dtype == np.float32:
            input_tensor = (input_tensor / 255.0).astype(np.float32)
        else:
            input_tensor = input_tensor.astype(in_dtype)

        interpreter.set_tensor(in_details[0]["index"], input_tensor)
        interpreter.invoke()

        boxes = interpreter.get_tensor(out_details[0]["index"])[0]
        classes = interpreter.get_tensor(out_details[1]["index"])[0].astype(np.int32)
        scores = interpreter.get_tensor(out_details[2]["index"])[0]

        # DEBUG: console-only detection output
        for cls, score in zip(classes, scores):
            if score > 0.2:
                print("DETECTION:", int(cls), float(score))

        # Find best person/dog detection above threshold (KEEPING YOUR EXACT LOGIC)
        best = None
        for cls, score, box in zip(classes, scores, boxes):
            if score < SCORE_THRESH:
                continue
            if cls in (COCO_PERSON, COCO_DOG):
                if best is None or score > best["score"]:
                    best = {"class_id": int(cls), "score": float(score), "box": box.tolist()}

        # Time accounting (based on current state)
        if best is not None:
            if best["class_id"] == COCO_DOG:
                dog_seconds += dt
            elif best["class_id"] == COCO_PERSON:
                person_seconds += dt

        # Publish detection (cooldown)
        if best and (now - last_publish_ts) >= COOLDOWN_SEC:
            label = "person" if best["class_id"] == COCO_PERSON else "dog"
            payload_ts = int(now)
            logging.info("DETECTED: %s score=%.3f", label, best["score"])

            try:
                sender.send_detection(label, best["score"], payload_ts)
                last_publish_ts = now
            except Exception as e:
                logging.error("Arduino Cloud send_detection failed: %r", e)

        # Publish "No detection" (also cooldown-ish, but separate timer so it doesn't override true detections)
        if (best is None) and ((now - last_no_det_publish) >= COOLDOWN_SEC):
            payload_ts = int(now)
            try:
                sender.send_detection("No detection", 0.0, payload_ts)
                last_no_det_publish = now
                logging.info("STATE: No detection")
            except Exception as e:
                logging.error("Arduino Cloud send_detection(No detection) failed: %r", e)

        # 6-hour report
        if now >= next_report_time:
            payload_ts = int(now)
            report_text = (
                f"report ( person detected : {fmt_hours(person_seconds)} hours , "
                f"dog detected : {fmt_hours(dog_seconds)} hours )"
            )
            logging.info("REPORT: %s", report_text)
            try:
                sender.send_report(report_text, payload_ts)
            except Exception as e:
                logging.error("Arduino Cloud send_report failed: %r", e)

            # Reset counters for next window (this makes each report cover the last 6 hours)
            dog_seconds = 0.0
            person_seconds = 0.0
            next_report_time = now + REPORT_EVERY_SEC

        time.sleep(0.05)


if __name__ == "__main__":
    main()
