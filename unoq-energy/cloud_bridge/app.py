import os
import json
import time
import threading
import logging
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from arduino_iot_cloud import ArduinoCloudClient

# -----------------------------
# Config (env)
# -----------------------------
MQTT_HOST = os.getenv("MQTT_HOST", "mqtt")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

# Comma-separated list in docker-compose:
# SUB_TOPICS: "coach/summary/top_consumers,coach/recipe/recommendations"
SUB_TOPICS_RAW = os.getenv(
    "SUB_TOPICS",
    "coach/summary/top_consumers,coach/recipe/recommendations",
)
SUB_TOPICS = [t.strip() for t in SUB_TOPICS_RAW.split(",") if t.strip()]

MIN_CLOUD_UPDATE_SEC = float(os.getenv("MIN_CLOUD_UPDATE_SEC", "30"))

ARDUINO_DEVICE_ID = os.getenv("ARDUINO_DEVICE_ID", "").strip()
ARDUINO_SECRET_KEY = os.getenv("ARDUINO_SECRET_KEY", "").strip()

# -----------------------------
# Cloud variables (must match Arduino Cloud EXACTLY)
# -----------------------------
CLOUD_VARS = {
    "coach_last_update_utc": "str",
    "coach_recommendation": "str",
    "coach_savings_next_week_usd": "float",
    "coach_top_circuit": "str",
    "coach_top_cost_today_usd": "float",
    "coach_top_kwh_today": "float",
}

# -----------------------------
# State
# -----------------------------
lock = threading.Lock()
latest_summary = None
latest_recipe = None

last_cloud_push_ts = 0.0
last_pushed_fingerprint = None


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s:%(name)s:%(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cloud-bridge")


# -----------------------------
# Helpers
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def normalize_str(x) -> str:
    if x is None:
        return ""
    return str(x)


# -----------------------------
# Arduino Cloud Sender
# -----------------------------
class ArduinoCloudSender:
    """
    Arduino IoT Cloud client with sync_mode=True + update loop.
    """

    def __init__(self):
        if not ARDUINO_DEVICE_ID or not ARDUINO_SECRET_KEY:
            raise RuntimeError(
                "Missing ARDUINO_DEVICE_ID / ARDUINO_SECRET_KEY env vars"
            )

        log.info("Connecting to Arduino IoT cloud...")
        self.client = ArduinoCloudClient(
            device_id=ARDUINO_DEVICE_ID,
            username=ARDUINO_DEVICE_ID,
            password=ARDUINO_SECRET_KEY,
            sync_mode=True,
        )

        self._register_all_vars()

        # Start the cloud client
        self.client.start()

        # Keep the client alive
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._update_loop, daemon=True)
        self._t.start()

    def _register_all_vars(self):
        """
        IMPORTANT: Without registering, cloud['var']=... can raise KeyError.
        We explicitly register ALL variables listed in CLOUD_VARS.
        """
        for name, kind in CLOUD_VARS.items():
            self._register_one(name, kind)

    def _register_one(self, name: str, kind: str):
        # Different library builds accept different register() signatures.
        # Try the common ones safely.
        try:
            # Most common: register(name)
            self.client.register(name)
            log.info("Registered cloud var: %s", name)
            return
        except TypeError:
            pass
        except Exception:
            # already registered or discovery related
            log.info("Registered cloud var: %s", name)
            return

        # Fallback: register(name, initial_value)
        try:
            init_val = "" if kind == "str" else 0.0
            self.client.register(name, init_val)
            log.info("Registered cloud var: %s", name)
        except Exception as e:
            log.warning("Could not register %s (%s): %s", name, kind, e)

    def set(self, key: str, value):
        """
        Safe setter:
        - casts types
        - if KeyError, tries register + retry once
        """
        kind = CLOUD_VARS.get(key, "str")
        if kind == "float":
            value = to_float(value, 0.0)
        else:
            value = normalize_str(value)

        try:
            self.client[key] = value
        except KeyError:
            # variable not in records -> register and retry once
            log.warning("Cloud var missing in records, registering now: %s", key)
            self._register_one(key, kind)
            self.client[key] = value

    def _update_loop(self):
        while not self._stop.is_set():
            try:
                # Some versions need update() to progress. If not present, ignore.
                if hasattr(self.client, "update"):
                    self.client.update()
            except Exception as e:
                log.warning("Cloud update loop error: %s", e)
            time.sleep(0.2)

    def stop(self):
        self._stop.set()


# -----------------------------
# Push logic (rate-limited)
# -----------------------------
def push_to_cloud(cloud: ArduinoCloudSender):
    global last_cloud_push_ts, last_pushed_fingerprint

    with lock:
        s = latest_summary
        r = latest_recipe

    if not s and not r:
        return

    now = time.time()
    if (now - last_cloud_push_ts) < MIN_CLOUD_UPDATE_SEC:
        return

    # Build a stable fingerprint to avoid spamming identical data
    fingerprint_obj = {
        "summary": s,
        "recipe": r,
    }
    fingerprint = json.dumps(fingerprint_obj, sort_keys=True)

    if fingerprint == last_pushed_fingerprint:
        return

    # Defaults
    top_circuit = ""
    top_kwh = 0.0
    top_cost = 0.0

    # Extract summary "top consumer"
    if isinstance(s, dict):
        # Prefer top_by_cost, then top_by_kwh
        candidates = []
        if isinstance(s.get("top_by_cost"), list) and s["top_by_cost"]:
            candidates = s["top_by_cost"]
        elif isinstance(s.get("top_by_kwh"), list) and s["top_by_kwh"]:
            candidates = s["top_by_kwh"]

        if candidates:
            top = candidates[0] or {}
            top_circuit = normalize_str(top.get("circuit", ""))
            top_kwh = to_float(top.get("kwh", 0.0), 0.0)
            top_cost = to_float(top.get("cost_usd", 0.0), 0.0)

    # Extract recipe/recommendation
    recommendation = ""
    savings = 0.0
    if isinstance(r, dict):
        recommendation = normalize_str(r.get("recommendation", ""))
        # Your coach publishes savings_week; cloud var is coach_savings_next_week_usd
        savings = to_float(r.get("savings_week", r.get("savings_next_week", 0.0)), 0.0)

    # Push to Arduino Cloud
    try:
        cloud.set("coach_top_circuit", top_circuit)
        cloud.set("coach_top_kwh_today", top_kwh)
        cloud.set("coach_top_cost_today_usd", top_cost)

        cloud.set("coach_recommendation", recommendation)
        cloud.set("coach_savings_next_week_usd", savings)

        cloud.set("coach_last_update_utc", utc_now_iso())

        last_cloud_push_ts = now
        last_pushed_fingerprint = fingerprint

        log.info(
            "PUSH OK -> top='%s' kwh=%.3f cost=%.3f rec='%s' savings=%.3f",
            top_circuit, top_kwh, top_cost, recommendation, savings
        )

    except Exception as e:
        log.exception("Cloud push failed: %s", e)


# -----------------------------
# MQTT callbacks
# -----------------------------
def on_connect(client, userdata, flags, rc):
    log.info("MQTT connected: %s", rc)
    for t in SUB_TOPICS:
        client.subscribe(t)
        log.info("Subscribed to %s", t)


def on_message(client, userdata, msg):
    global latest_summary, latest_recipe
    cloud = userdata["cloud"]

    try:
        payload_raw = msg.payload.decode("utf-8", errors="replace").strip()
        payload = json.loads(payload_raw) if payload_raw else {}

        log.info("Message on %s: %s", msg.topic, payload)

        with lock:
            if msg.topic.endswith("/top_consumers"):
                latest_summary = payload
            elif msg.topic.endswith("/recommendations"):
                latest_recipe = payload

        push_to_cloud(cloud)

    except Exception:
        log.exception("Processing error")


# -----------------------------
# Main
# -----------------------------
def main():
    cloud = ArduinoCloudSender()

    m = mqtt.Client(userdata={"cloud": cloud})
    m.on_connect = on_connect
    m.on_message = on_message

    while True:
        try:
            log.info("Connecting to MQTT broker %s:%d ...", MQTT_HOST, MQTT_PORT)
            m.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            m.loop_forever(retry_first_connection=True)
        except Exception as e:
            log.warning("MQTT loop error: %s (retrying in 3s)", e)
            time.sleep(3)


if __name__ == "__main__":
    main()
