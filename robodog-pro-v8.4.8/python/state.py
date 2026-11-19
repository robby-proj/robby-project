"""
state.py - Robodog Pro v8.4.6 shared configuration and runtime state.

This module is the single in-memory state contract shared by the WebUI,
RouterBridge boundary, camera and object-detection pipeline, autonomous-control
worker, gamepad safety subsystem, audio worker, OLED publisher, LED-matrix
queue, hardware precheck, status logger, and watchdog.

Design rules
------------
* No camera capture, annotated image, controller event, telemetry sample, or
  application status is persisted to disk.
* Camera data is held in one replaceable RAM frame slot and optional base64 API
  fields only.
* Locks protect shared mutable objects that cross worker-thread boundaries.
* Bounded queues prevent display or feedback producers from creating unlimited
  backlog when RouterBridge or an output device is slow.
* HOST_IP, when supplied by App Lab, is the preferred operator-facing address.
  STATIC_FALLBACK_IP is used only when HOST_IP is unavailable.
* AP switching remains a host NetworkManager operation; the application
  container stores and reports AP metadata but does not control host wlan0.
* Subsystems own their detailed dictionaries, while STATE exposes a coordinated
  top-level snapshot to diagnostics and the WebUI.

Ownership summary
-----------------
CAMERA_STATE        perception.py
BRIDGE_STATE        hw_bridge.py
LIVE_DETECTIONS     perception.py and autonomy.py
HARDWARE            hw_bridge.py hardware precheck
MOTION              hw_bridge.py motion worker
MOOD                autonomy.py and hw_bridge.py
STATE['controller']  input_controller.py
STATE['sequence']    autonomy.py
STATE['autonomy']    autonomy.py
STATE['wifi_status'] ui_api.py
THREAD_*             main.py and hw_bridge.py watchdog
"""

import os
import queue
import threading
import time

# --------------------------------------------------------------------------
# Release identity
# --------------------------------------------------------------------------
APP_VERSION = "robodog-2026.08.19-v8.4.6-release-candidate"

# --------------------------------------------------------------------------
# Operator-facing network identity
# --------------------------------------------------------------------------
# The App Lab Python app runs inside a Docker container. From inside that
# container it can only see its own bridge address (172.16-31.x) - NOT the
# host's real wlan0 IP. So auto-detection returns a useless 172.x address.
# ADVERTISED_IP is the address the operator actually connects to and is what
# the matrix/OLED display. App Lab injects HOST_IP for normal client mode.
# The fallback is used only if HOST_IP is unavailable and must be declared
# before ADVERTISED_IP is evaluated. AP mode uses 10.42.0.1 when the host AP
# profile is active.
STATIC_FALLBACK_IP = "192.168.7.149"
ADVERTISED_IP = os.environ.get("HOST_IP", "").strip() or STATIC_FALLBACK_IP

# --------------------------------------------------------------------------
# WiFi AP (host-level profile; the App Lab container cannot run nmcli)
# --------------------------------------------------------------------------
AP_SSID         = "DOGZILLA-UNOQ"
AP_PASSWORD     = "dogzilla123"
AP_INTERFACE    = "wlan0"
AP_URL          = "http://10.42.0.1:7000"
AP_PROFILE_NAME = "dogzilla-ap"
AP_STATIC_IP    = "10.42.0.1"
AP_FALLBACK_ENABLED = True
AP_FALLBACK_DELAY_SEC = 120.0
AP_FALLBACK_POLL_SEC = 2.0
AP_HOST_SCRIPT = "/home/arduino/ArduinoApps/robodog-pro-v8-4-6/host/dogzilla-ap.sh"

# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------
CAMERA_SEARCH_RANGE      = range(0, 8)
CAMERA_ROTATE_180        = True
CAMERA_FRAME_INTERVAL    = 0.0
##CAMERA_FRAME_INTERVAL    = 0.15
CAMERA_STALL_TIMEOUT_SEC = 15

# --------------------------------------------------------------------------
# Object detection
# --------------------------------------------------------------------------
CONFIDENCE                    = 0.25
CONTINUOUS_DETECTION_ENABLED  = True
CONTINUOUS_DETECTION_INTERVAL = 0.20
#CONTINUOUS_DETECTION_INTERVAL = 0.4
CONTINUOUS_DETECTION_TARGET   = "person"
LIVE_OVERLAY_MAX_AGE_SEC      = 3.0

# --------------------------------------------------------------------------
# Display cadence / battery hysteresis
# --------------------------------------------------------------------------
MATRIX_SEARCHING_MIN_INTERVAL = 6.0
OLED_IP_REFRESH_SEC           = 30.0
OLED_STATUS_REFRESH_SEC       = 10.0
LOW_BATTERY_PCT               = 20
LOW_BATTERY_RECOVERY_PCT      = 25

# --------------------------------------------------------------------------
# Autonomous IMU wander and recovery
# --------------------------------------------------------------------------
# The camera is often unavailable on this hardware, so autonomous mode is
# driven by the Modulino IMU (accelerometer + derived vibration) instead of
# vision. The dog wanders forward, turns periodically, and reacts to bumps
# (vibration spikes) and tilt (accel) reported over the Bridge.
AUTONOMY_TICK_SEC          = 0.2    # control-loop period
AUTONOMY_FORWARD_SEC       = 3.0    # walk forward before a scheduled turn
AUTONOMY_TURN_SEC          = 1.2    # duration of a scheduled turn
AUTONOMY_IMPACT_VIBRATION  = 1.20   # vibration above this = bump/stuck -> recover
AUTONOMY_TILT_G            = 0.70   # raised for mounted Movement sensor / walking shocks
AUTONOMY_SAFETY_CONFIRM_SAMPLES = 5
AUTONOMY_SAFETY_CLEAR_SAMPLES   = 8
AUTONOMY_SAFETY_ONLY_WHILE_ACTIVE = True
VISION_AUTONOMY_FRAME_MAX_AGE_SEC = 1.2
AUTONOMY_OWNER_MANUAL = "manual"
AUTONOMY_OWNER_IMU = "imu"
AUTONOMY_OWNER_VISION = "vision"
AUTONOMY_OWNER_SEQUENCE = "sequence"
AUTONOMY_OWNER_SAFETY = "safety"   # |x| or |y| accel above this = tipped -> recover
AUTONOMY_RECOVER_BACK_SEC  = 1.0    # back up time after an impact
AUTONOMY_SNIFF_EVERY_SEC   = 20.0   # occasional dog-like flourish while wandering

# --------------------------------------------------------------------------
# Watchdog / boot
# --------------------------------------------------------------------------
THREAD_HEARTBEAT_STALE_SEC = 45.0
BOOT_PROBE_INTERVAL        = 0.5
BOOT_PROBE_TIMEOUT         = 12.0

# --------------------------------------------------------------------------
# System paths (read-only probes)
# --------------------------------------------------------------------------
USB_MODE_FILE = "/sys/kernel/debug/usb/4e00000.usb/mode"

# --------------------------------------------------------------------------
# Locks
# --------------------------------------------------------------------------
state_lock            = threading.Lock()
bridge_lock           = threading.RLock()
live_detections_lock  = threading.Lock()
live_frame_lock       = threading.Lock()

# --------------------------------------------------------------------------
# In-memory frame buffer (single-slot, overwritten each cycle - no disk)
# --------------------------------------------------------------------------
LIVE_FRAME_RGB = None

# --------------------------------------------------------------------------
# Bounded matrix event queue (decouples callers from Bridge latency)
# --------------------------------------------------------------------------
MATRIX_QUEUE = queue.Queue(maxsize=32)

# --------------------------------------------------------------------------
# Watchdog registry
# --------------------------------------------------------------------------
THREAD_HEARTBEATS = {}
THREAD_TARGETS    = {}

# --------------------------------------------------------------------------
# Owned state dicts
# --------------------------------------------------------------------------
CAMERA_STATE = {
    "ok": False, "running": False,
    "frame_base64": None,
    "timestamp": 0, "fps": 0,
    "error": None, "index": None, "reopens": 0,
}

BRIDGE_STATE = {
    "ok": False, "last_ok": 0,
    "last_error": None, "consecutive_errors": 0,
}

LIVE_DETECTIONS = {
    "detections": [], "timestamp": 0, "processing_time_ms": 0,
    "count": 0, "target": CONTINUOUS_DETECTION_TARGET,
    "target_count": 0, "found_target": False, "confidence": 0.0,
    "error": None,
    "_raw_results": None,
    "_last_matrix_search_ms": 0,
    "_prev_found_target": False,
    "_first_result_dumped": False,
    "_overlay_err_logged": False,
}

HARDWARE = {
    "probed": False, "probe_error": None,
    "movement_present": None, "movement_responds": None,
    "movement_addr": None, "dog_battery_probe": None,
    "dog_firmware": None,
    "buzzer_present": None,
    "eyes": {
        "present": False, "mode": None,
        "flush_ok_count": None, "flush_error_count": None,
        "bus_fault": None, "updated": 0,
    },
    "bridge_ok": False,
}

MOTION = {
    "ok": False, "present": False, "responds": False,
    "addr": 0, "x": 0.0, "y": 0.0, "z": 0.0,
    "vibration": 0.0, "moving": False, "still_seconds": 0.0,
    "last_nonzero_ts": 0.0,
}

MOOD = {
    "current": "normal",
    "low_battery_latched": False,
}

STATE = {
    "version": APP_VERSION,
    "release": "8.4.6",
    "captures_persisted": False,
    "image_storage": "RAM-only",
    "mode": "manual",
    "last_command": "none",
    "last_action": 0,
    "gamepad": "not connected",
    "gamepad_enabled": True,
    # Controller diagnostics (input_controller.py owns this)
    "controller": {
        "enabled": True,
        "receiver_present": False,
        "controller_active": False,
        "connected": False,
        "device": None,
        "api": None,
        "name": None,
        "permission_ok": None,
        "last_event": None,
        "last_event_time": 0,
        "last_command": "stop",
        "desired_command": "stop",
        "events": 0,
        "init_events": 0,
        "reconnects": 0,
        "disconnects": 0,
        "armed": False,
        "hard_stop_latched": True,
        "owner": "stopped",
        "bridge_available": False,
        "safety_blocked": False,
        "safety_reason": None,
        "reader_active": False,
        "reader_generation": 0,
        "command_worker_alive": False,
        "axis_state": {
            "x": 0.0, "y": 0.0, "turn": 0.0,
            "right_y": 0.0, "hat_x": 0.0, "hat_y": 0.0,
        },
        "mapping": "shanwan-2563-0526-ps2-v8.4",
        "error": None,
    },
    # Dog-walkaround sequence + autonomous status (autonomy.py owns these)
    "sequence": {"running": False, "name": None, "step": "idle", "error": None},
    "autonomy": {"active": False, "phase": "idle", "reason": None,
               "owner": "manual", "requested_owner": "imu",
                 "impacts": 0, "vibration": 0.0},
    "wifi_ap": "standby",
    "wifi_ssid": AP_SSID,
    "wifi_password": AP_PASSWORD,
    "wifi_url": AP_URL,
    "wifi_status": {
        "ok": False, "mode": "unknown", "ssid": None, "ip": None,
        "profile": None, "client_count": 0,
        "nmcli_available": False, "updated": 0,
        "source": "container-observation", "error": None,
    },
    "wifi_fallback": {
        "enabled": AP_FALLBACK_ENABLED,
        "delay_sec": AP_FALLBACK_DELAY_SEC,
        "poll_sec": AP_FALLBACK_POLL_SEC,
        "state": "pending", "started_at": 0, "deadline": 0,
        "remaining_sec": AP_FALLBACK_DELAY_SEC, "completed_at": 0,
        "client_connected": False, "client_ssid": None, "client_ip": None,
        "decision": "waiting", "reason": "startup pending",
        "ap_requested": False, "ap_started": False,
        "ap_start_attempts": 0, "ap_start_returncode": None,
        "active_mode": "unknown", "active_url": None,
        "last_error": None, "updated": 0,
    },
    "target_object": "person",
    "language": "English",
    "camera_rotate_180": CAMERA_ROTATE_180,
    "continuous_detection": {
        "enabled": CONTINUOUS_DETECTION_ENABLED,
        "target": CONTINUOUS_DETECTION_TARGET,
        "interval_sec": CONTINUOUS_DETECTION_INTERVAL,
    },
    "vibration": {"ok": False, "x": 0, "y": 0, "z": 0, "vibration": 0},
    "telemetry": {
        "ok": False, "x": 0, "y": 0, "z": 0, "vibration": 0,
        "battery": None, "roll": None, "pitch": None, "yaw": None,
    },
    "last_detection": {
        "ok": False, "target": None, "found": False, "count": 0,
        "processing_time_ms": None,
        "image_base64": None, "detections": [],
    },
    "hardware": HARDWARE,
    "motion": MOTION,
    "mood": MOOD,
    "camera": CAMERA_STATE,
    "bridge": BRIDGE_STATE,
    "audio": {
        "speaker_available": False, "muted": False, "current": None,
        "last_event": None, "queue_depth": 0, "error": None,
    },
    "eyes_motion_reactive": True,
    "chat": [{"role": "robot",
              "text": "Robodog Pro v8.4.6 ready. Safe motion ownership, recovered OLED eyes, person feedback, displays, and audio are available."}],
}

START_TIME = time.time()


def heartbeat(name: str) -> None:
    """Record a monotonic heartbeat for one supervised worker."""
    THREAD_HEARTBEATS[str(name)] = time.monotonic()


def robot_say(text: object) -> None:
    """Print and append one bounded in-memory robot chat message."""
    message = str(text)
    print("[ROBOT]", message, flush=True)
    with state_lock:
        STATE["chat"].append({"role": "robot", "text": message})
        STATE["chat"] = STATE["chat"][-40:]
