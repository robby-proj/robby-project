import os
import cv2
import time
import json
import queue
import signal
import logging
import threading
import subprocess
from datetime import datetime
from flask import Flask, jsonify, render_template, send_from_directory

# -----------------------------
# Config
# -----------------------------
CAMERA_INDEX = int(os.environ.get("USB_CAMERA_INDEX", "0"))
CAPTURE_SECONDS = float(os.environ.get("CAPTURE_SECONDS", "3.0"))
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", "480"))

YOLO_ONNX_PATH = os.environ.get("YOLO_ONNX_PATH", "/models/coco/yolov8n.onnx")
COCO_NAMES_PATH = os.environ.get("COCO_NAMES_PATH", "/models/coco/coco.names")
PERSON_CONFIDENCE = float(os.environ.get("PERSON_CONFIDENCE", "0.45"))
NMS_THRESHOLD = float(os.environ.get("NMS_THRESHOLD", "0.45"))
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", "640"))

YZMA_DIR = os.environ.get("YZMA_DIR", "/yzma")
YZMA_IMAGES_DIR = os.environ.get("YZMA_IMAGES_DIR", "/data/images")
VLM_MODEL_PATH = os.environ.get("VLM_MODEL_PATH", "/models/SmolVLM2-500M-Video-Instruct-Q8_0.gguf")
VLM_MMPROJ_PATH = os.environ.get("VLM_MMPROJ_PATH", "/models/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf")
VLM_PROMPT = os.environ.get(
    "VLM_PROMPT",
    "Look at the person in this image and identify the main color of the top clothing and the main color of the bottom clothing. Reply in one short sentence."
)

LOG_FILE = os.environ.get("LOG_FILE", "/data/logs/vision_web.log")
STATE_FILE = os.environ.get("STATE_FILE", "/data/state/vision_web_state.json")
TRIGGER_FILE = os.environ.get("TRIGGER_FILE", "/host_tmp/unoq_trigger_detection")
MIN_SECONDS_BETWEEN_TRIGGERS = float(os.environ.get("MIN_SECONDS_BETWEEN_TRIGGERS", "5.0"))

# Working USB speaker path you validated on the host
AUDIO_PLAYBACK_DEVICE = os.environ.get("AUDIO_PLAYBACK_DEVICE", "plughw:2,0")
TTS_VOICE = os.environ.get("TTS_VOICE", "en-us")
TTS_SPEED = os.environ.get("TTS_SPEED", "155")

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
os.makedirs(YZMA_IMAGES_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

app = Flask(__name__)
trigger_queue = queue.Queue()
stop_event = threading.Event()
busy_lock = threading.Lock()
last_trigger_time = 0.0

state = {
    "status": "idle",
    "last_event_time": None,
    "last_image": None,
    "last_debug_image": None,
    "last_full_image": None,
    "last_result": None,
    "chat_messages": [
        {
            "role": "system",
            "text": "System ready. Waiting for trigger from App Lab/Linux helper."
        }
    ]
}


# -----------------------------
# Helpers
# -----------------------------
def save_state():
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def update_status(new_status: str):
    state["status"] = new_status
    state["last_event_time"] = datetime.now().isoformat()
    save_state()


def add_chat(role: str, text: str):
    state["chat_messages"].append({
        "role": role,
        "text": text,
        "time": datetime.now().strftime("%H:%M:%S")
    })
    state["chat_messages"] = state["chat_messages"][-30:]
    save_state()


def current_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_coco_names(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"COCO names file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# -----------------------------
# Detector
# -----------------------------
class YoloPersonDetector:
    def __init__(self, onnx_path, coco_names_path, conf_thresh, nms_thresh, input_size):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"YOLO ONNX file not found: {onnx_path}")

        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.class_names = load_coco_names(coco_names_path)

        if "person" not in self.class_names:
            raise RuntimeError("'person' class not found in COCO names file")

        self.person_class_id = self.class_names.index("person")
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size

    def preprocess(self, image):
        return cv2.dnn.blobFromImage(
            image,
            scalefactor=1 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False
        )

    def detect_best_person(self, frame):
        h, w = frame.shape[:2]

        blob = self.preprocess(frame)
        self.net.setInput(blob)
        outputs = self.net.forward()

        out = outputs
        if len(out.shape) == 3 and out.shape[1] < out.shape[2]:
            out = out[0].T
        elif len(out.shape) == 3:
            out = out[0]

        boxes = []
        confidences = []

        for row in out:
            x, y, bw, bh = row[0], row[1], row[2], row[3]
            scores = row[4:]
            class_id = int(scores.argmax())
            conf = float(scores[class_id])

            if class_id != self.person_class_id or conf < self.conf_thresh:
                continue

            x_scale = w / float(self.input_size)
            y_scale = h / float(self.input_size)

            left = int((x - bw / 2) * x_scale)
            top = int((y - bh / 2) * y_scale)
            width = int(bw * x_scale)
            height = int(bh * y_scale)

            left = max(0, left)
            top = max(0, top)
            width = min(width, w - left)
            height = min(height, h - top)

            if width <= 0 or height <= 0:
                continue

            boxes.append([left, top, width, height])
            confidences.append(conf)

        if not boxes:
            return None

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.nms_thresh)
        if indices is None or len(indices) == 0:
            return None

        best = None
        best_score = -1.0

        for idx in indices:
            i = int(idx[0]) if hasattr(idx, "__len__") else int(idx)
            box = boxes[i]
            conf = confidences[i]
            area = box[2] * box[3]
            score = conf * area

            if score > best_score:
                best_score = score
                best = {
                    "box": box,
                    "confidence": conf,
                    "score": score
                }

        return best


# -----------------------------
# Camera and yzma
# -----------------------------
def capture_best_frame(detector):
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    start = time.time()
    best_frame = None
    best_meta = None

    try:
        while time.time() - start < CAPTURE_SECONDS:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            result = detector.detect_best_person(frame)
            if result is None:
                continue

            if best_meta is None or result["score"] > best_meta["score"]:
                best_frame = frame.copy()
                best_meta = result

        return best_frame, best_meta
    finally:
        cap.release()


def crop_person(frame, meta, padding=0.08):
    x, y, w, h = meta["box"]
    fh, fw = frame.shape[:2]

    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(fw, x + w + pad_x)
    y2 = min(fh, y + h + pad_y)

    return frame[y1:y2, x1:x2]


def save_frame_and_debug(frame, meta):
    ts = current_ts()

    crop = crop_person(frame, meta)

    image_path = os.path.join(YZMA_IMAGES_DIR, f"person_{ts}.jpg")
    full_path = os.path.join(YZMA_IMAGES_DIR, f"person_full_{ts}.jpg")
    debug_path = os.path.join(YZMA_IMAGES_DIR, f"person_{ts}_debug.jpg")

    cv2.imwrite(image_path, crop)
    cv2.imwrite(full_path, frame)

    dbg = frame.copy()
    x, y, w, h = meta["box"]
    cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        dbg,
        f"person {meta['confidence']:.2f}",
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )
    cv2.imwrite(debug_path, dbg)

    latest_crop = os.path.join(YZMA_IMAGES_DIR, "latest_person.jpg")
    latest_full = os.path.join(YZMA_IMAGES_DIR, "latest_person_full.jpg")
    latest_debug = os.path.join(YZMA_IMAGES_DIR, "latest_person_debug.jpg")

    cv2.imwrite(latest_crop, crop)
    cv2.imwrite(latest_full, frame)
    cv2.imwrite(latest_debug, dbg)

    return image_path, full_path, debug_path, latest_crop, latest_full, latest_debug


def run_yzma_vlm(image_path):
    if not os.path.isdir(YZMA_DIR):
        raise FileNotFoundError(f"YZMA_DIR not found: {YZMA_DIR}")
    if not os.path.exists(VLM_MODEL_PATH):
        raise FileNotFoundError(f"VLM model file not found: {VLM_MODEL_PATH}")
    if not os.path.exists(VLM_MMPROJ_PATH):
        raise FileNotFoundError(f"VLM mmproj file not found: {VLM_MMPROJ_PATH}")

    yzma_lib_path = os.path.join(YZMA_DIR, "lib")
    if not os.path.isdir(yzma_lib_path):
        raise FileNotFoundError(f"yzma lib directory not found: {yzma_lib_path}")

    cmd = [
        "go", "run", "./examples/vlm/",
        "-model", VLM_MODEL_PATH,
        "-mmproj", VLM_MMPROJ_PATH,
        "-lib", yzma_lib_path,
        "-image", image_path,
        "-p", VLM_PROMPT,
    ]

    logging.info("Running yzma command: %s", " ".join(cmd))

    proc = subprocess.run(
        cmd,
        cwd=YZMA_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300
    )

    logging.info("yzma stdout: %s", proc.stdout)
    logging.info("yzma stderr: %s", proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(f"yzma failed: {proc.stderr}\n{proc.stdout}")

    output = proc.stdout.strip()

    if output.startswith("Usage: vlm"):
        raise RuntimeError(f"yzma returned usage text instead of an answer:\n{output}")

    return output


def speak_text(text):
    safe_text = text.strip()
    if not safe_text:
        return

    # Keep the sentence friendly for speech
    logging.info("Speaking text: %s", safe_text)

    cmd = f'espeak-ng -v {TTS_VOICE} -s {TTS_SPEED} "{safe_text}" --stdout | aplay -D {AUDIO_PLAYBACK_DEVICE}'

    proc = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60
    )

    logging.info("tts stdout: %s", proc.stdout)
    logging.info("tts stderr: %s", proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(f"TTS failed: {proc.stderr}")


# -----------------------------
# Workflow
# -----------------------------
def process_trigger(detector):
    global last_trigger_time

    with busy_lock:
        now = time.time()
        if now - last_trigger_time < MIN_SECONDS_BETWEEN_TRIGGERS:
            logging.info("Trigger ignored due to cooldown")
            return

        last_trigger_time = now

        update_status("detecting")
        add_chat("user", "Trigger received. Start person detection.")

        try:
            frame, meta = capture_best_frame(detector)

            if frame is None or meta is None:
                update_status("no_person_detected")
                state["last_result"] = "No person detected."
                add_chat("assistant", "No person was detected in the capture window.")
                save_state()
                return

            update_status("person_detected")
            image_path, full_path, debug_path, latest_crop, latest_full, latest_debug = save_frame_and_debug(frame, meta)

            state["last_image"] = os.path.basename(latest_crop)
            state["last_debug_image"] = os.path.basename(latest_debug)
            state["last_full_image"] = os.path.basename(latest_full)
            save_state()

            add_chat("system", "Person detected. Running VLM clothing-color analysis.")
            update_status("running_vlm")

            vlm_result = run_yzma_vlm(image_path)

            state["last_result"] = vlm_result
            update_status("done")
            add_chat("assistant", vlm_result)
            save_state()

            speak_text(vlm_result)

        except Exception as e:
            logging.exception("Workflow failed")
            update_status("error")
            state["last_result"] = str(e)
            add_chat("assistant", f"Error: {e}")
            save_state()


# -----------------------------
# Trigger file watcher
# -----------------------------
def trigger_file_watcher():
    logging.info("Watching trigger file: %s", TRIGGER_FILE)

    while not stop_event.is_set():
        try:
            if os.path.exists(TRIGGER_FILE):
                logging.info("Trigger file detected")
                try:
                    os.remove(TRIGGER_FILE)
                except FileNotFoundError:
                    pass
                trigger_queue.put(True)

            time.sleep(0.2)

        except Exception as e:
            logging.warning("Trigger watcher retrying: %s", e)
            time.sleep(1)


def worker(detector):
    while not stop_event.is_set():
        try:
            trigger_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        process_trigger(detector)


# -----------------------------
# Flask routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    return jsonify(state)


@app.route("/images/<path:filename>")
def images(filename):
    return send_from_directory(YZMA_IMAGES_DIR, filename)


@app.route("/api/manual_trigger", methods=["POST"])
def api_manual_trigger():
    trigger_queue.put(True)
    return jsonify({"ok": True})


# -----------------------------
# Main
# -----------------------------
def handle_signal(sig, frame):
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    detector = YoloPersonDetector(
        YOLO_ONNX_PATH,
        COCO_NAMES_PATH,
        PERSON_CONFIDENCE,
        NMS_THRESHOLD,
        INPUT_SIZE
    )

    save_state()

    t1 = threading.Thread(target=trigger_file_watcher, daemon=True)
    t2 = threading.Thread(target=worker, args=(detector,), daemon=True)

    t1.start()
    t2.start()

    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)


if __name__ == "__main__":
    main()
