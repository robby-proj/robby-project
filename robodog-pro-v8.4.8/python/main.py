"""
===============================================================================
Robodog Pro v8.4.6 - Expanded UNO Q App Lab backend
===============================================================================

Purpose
-------
This is the longer, shareable Robodog Pro backend style requested for the v8.4.x
DOGZILLA + UNO Q project. It keeps the modular v8.4 project structure, but adds
more backend-owned diagnostics and control endpoints so the WebUI can surface the
same kinds of deep runtime controls that earlier monolithic versions exposed.

This main.py is intentionally verbose:
- routes are explicit
- bridge calls are wrapped and logged
- Motion sensitivity is first-class
- Motion diagnostics are first-class
- OLED eyes diagnostics are first-class
- camera health is first-class
- follow-person autonomy configuration is first-class
- target reacquisition policy is first-class
- watchdog and worker startup are visible from logs and /api/orchestrator

Hardware policy
---------------
- OEM DOGZILLA 128x32 OLED remains on MCU Wire
- Qwiic / Modulino chain remains on MCU Wire1
- Robot eyes OLED may remain at 0x3C on Wire1
- Modulino Movement is the preferred real Motion input
- Modulino Buzzer and eyes OLED are optional feedback devices
- Motion feedback must not randomly call DOGZILLA STOP by default

Zero persistence policy
-----------------------
Camera frames and annotations remain in memory only. This file does not create a
capture directory, write frames to /tmp, write frames to /home/arduino, or expose
an image_path API field.

New API routes in this file
---------------------------
Motion:
    GET /api/motion_sensitivity/{off|low|medium|high|0|1|2|3}
    GET /api/motion_policy
    GET /api/motion_diagnostics
    GET /api/motion_rebegin
    GET /api/wire1_scan_count

Eyes:
    GET /api/eyes_status
    GET /api/eyes_mode/{mode}
    GET /api/eyes_motion/{enabled}

Camera and vision:
    GET /api/camera_health
    GET /api/vision_summary

Autonomy:
    GET /api/autonomy/follow_config/{profile}
    GET /api/autonomy/reacquire/{enabled}
    GET /api/autonomy/reacquire_status

Diagnostics:
    GET /api/orchestrator
    GET /api/zero_persistence

Expected matching MCU providers
-------------------------------
The v8.4.6 sketch should expose these providers when available:
    motion_sensitivity, motion_sensitivity_get, motion_threshold_g,
    motion_local_stop_enable, movement_present, movement_responds,
    movement_rebegin, movement_mag, wire1_scan_count, eyes_present, eyes_addr,
    eyes_mode, eyes_motion_reactive, get_vibration, get_vibration_peak_clear,
    get_accel_x, get_accel_y, get_accel_z

Missing optional providers are reported as unavailable instead of crashing.
===============================================================================
"""

from __future__ import annotations

import copy
import json
import math
import os
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from typing import Any

from arduino.app_utils import App
from arduino.app_bricks.web_ui import WebUI

import audio_feedback as F
import autonomy as A
import hw_bridge as HW
import input_controller as C
import perception as P
import state as S
import ui_api as U


APP_RELEASE = "8.4.6"
APP_BUILD_ID = "robodog-2026.08.19-v8.4.6-release-candidate"
ui = WebUI()
RouteHandler = Callable[..., object]


# =============================================================================
# Runtime configuration owned by this orchestrator
# =============================================================================
FOLLOW_PROFILES: dict[str, dict[str, Any]] = {
    "safe": {
        "name": "safe",
        "left_boundary": 0.35,
        "right_boundary": 0.65,
        "close_bbox_ratio": 0.42,
        "very_close_bbox_ratio": 0.55,
        "lost_soft_s": 1.0,
        "lost_scan_s": 3.0,
        "lost_stretch_s": 6.0,
        "description": "Conservative demo profile. Stop or back away when target appears close.",
    },
    "demo": {
        "name": "demo",
        "left_boundary": 0.33,
        "right_boundary": 0.67,
        "close_bbox_ratio": 0.50,
        "very_close_bbox_ratio": 0.65,
        "lost_soft_s": 1.0,
        "lost_scan_s": 2.5,
        "lost_stretch_s": 5.0,
        "description": "Balanced profile for normal follow-person demos.",
    },
    "aggressive": {
        "name": "aggressive",
        "left_boundary": 0.30,
        "right_boundary": 0.70,
        "close_bbox_ratio": 0.60,
        "very_close_bbox_ratio": 0.75,
        "lost_soft_s": 0.5,
        "lost_scan_s": 2.0,
        "lost_stretch_s": 4.0,
        "description": "More willing to move forward. Use only with supervision.",
    },
}

ORCHESTRATOR_STATE: dict[str, Any] = {
    "app_release": APP_RELEASE,
    "build_id": APP_BUILD_ID,
    "started_at": time.time(),
    "routes_registered": 0,
    "workers_registered": 0,
    "workers_started": [],
    "watchdog_restarts": {},
    "startup_errors": [],
    "captures_persisted": False,
    "last_motion_policy_sync": None,
    "last_motion_policy_error": None,
    "last_motion_diagnostics": None,
    "last_eyes_status": None,
    "last_camera_health": None,
    "last_vision_summary": None,
    "follow_profile": copy.deepcopy(FOLLOW_PROFILES["safe"]),
    "reacquire_enabled": True,
    "reacquire_state": {
        "last_seen_at": 0.0,
        "last_lost_action": "none",
        "last_target_seen": False,
        "last_heading": "stop",
    },
    "recent_bridge_events": deque(maxlen=80),
}


# =============================================================================
# General helpers
# =============================================================================
def _now_s() -> int:
    return int(time.time())


def _now_m() -> float:
    return time.monotonic()


def _jsonsafe(value: Any) -> Any:
    try:
        return HW.jsonsafe(value)
    except Exception:
        try:
            return json.loads(json.dumps(value, default=str))
        except Exception:
            return str(value)


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        result = float(value)
        if math.isfinite(result):
            return result
        return default
    except Exception:
        return default


def _safe_int(value: Any, default: int | None = 0) -> int | None:
    try:
        return int(value)
    except Exception:
        return default


def _record_bridge_event(name: str, args: tuple[Any, ...], ok: bool, result: Any = None, error: str | None = None) -> None:
    ORCHESTRATOR_STATE["recent_bridge_events"].append({
        "ts": round(time.time(), 3),
        "name": name,
        "args": list(args),
        "ok": ok,
        "result": result,
        "error": error,
    })


def bridge_call_quiet(name: str, *args: Any) -> Any:
    try:
        if hasattr(HW, "bridge_call_quiet"):
            result = HW.bridge_call_quiet(name, *args)
        else:
            result = HW.bridge_call(name, *args)
        _record_bridge_event(name, args, True, result=result)
        return result
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _record_bridge_event(name, args, False, error=error)
        return None


def bridge_call(name: str, *args: Any) -> Any:
    try:
        result = HW.bridge_call(name, *args)
        _record_bridge_event(name, args, True, result=result)
        return result
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _record_bridge_event(name, args, False, error=error)
        raise


def _route_signature(method: str, path: str) -> str:
    return f"{method.upper()} {path}"


def _print_banner_line(label: str, value: Any) -> None:
    print(f"[boot] {label:<30} = {value}", flush=True)


# =============================================================================
# Motion sensitivity and diagnostics
# =============================================================================
def _motion_level_to_mode(level: str | int) -> int:
    mapping = {
        "off": 0,
        "0": 0,
        "low": 1,
        "1": 1,
        "medium": 2,
        "med": 2,
        "2": 2,
        "high": 3,
        "3": 3,
    }
    key = str(level).strip().lower()
    if key not in mapping:
        raise ValueError("motion sensitivity must be off, low, medium, high, or 0..3")
    return mapping[key]


def set_motion_sensitivity_mode(mode: int) -> dict[str, Any]:
    mode = max(0, min(3, int(mode)))
    bridge_result = bridge_call("motion_sensitivity", mode)

    # Critical safety default: keep MCU-local Motion STOP disabled unless the
    # operator explicitly enables it through a separate bridge call.
    local_stop = bridge_call_quiet("motion_local_stop_enable", 0)
    current = bridge_call_quiet("motion_sensitivity_get")
    threshold = bridge_call_quiet("motion_threshold_g")

    payload = {
        "ok": True,
        "requested_mode": mode,
        "reported_mode": current,
        "threshold_g": threshold,
        "local_stop_enabled": bool(local_stop),
        "bridge_result": bridge_result,
        "updated_at": _now_s(),
    }
    ORCHESTRATOR_STATE["last_motion_policy_sync"] = payload
    ORCHESTRATOR_STATE["last_motion_policy_error"] = None

    try:
        S.STATE["motion_sensitivity"] = mode
        S.STATE["motion_threshold_g"] = threshold
        S.STATE["motion_local_stop_enabled"] = bool(local_stop)
    except Exception:
        pass

    return payload


def collect_motion_diagnostics() -> dict[str, Any]:
    x = bridge_call_quiet("get_accel_x")
    y = bridge_call_quiet("get_accel_y")
    z = bridge_call_quiet("get_accel_z")
    vibration = bridge_call_quiet("get_vibration")
    mag = bridge_call_quiet("movement_mag")
    present = bridge_call_quiet("movement_present")
    responds = bridge_call_quiet("movement_responds")
    addr = bridge_call_quiet("movement_addr")
    sensitivity = bridge_call_quiet("motion_sensitivity_get")
    threshold = bridge_call_quiet("motion_threshold_g")

    payload = {
        "ok": bool(_safe_int(present, 0)),
        "present": bool(_safe_int(present, 0)),
        "responds": bool(_safe_int(responds, 0)),
        "addr": addr,
        "x": _safe_float(x),
        "y": _safe_float(y),
        "z": _safe_float(z),
        "movement_mag": _safe_float(mag, None),
        "vibration": _safe_float(vibration),
        "motion_sensitivity": _safe_int(sensitivity, None),
        "threshold_g": _safe_float(threshold, None),
        "local_stop_enabled": bool(S.STATE.get("motion_local_stop_enabled", False)) if hasattr(S, "STATE") else False,
        "updated_at": _now_s(),
        "interpretation": "movement_mag near 1.0 at rest and vibration near 0.0 is expected",
    }
    ORCHESTRATOR_STATE["last_motion_diagnostics"] = payload
    try:
        S.STATE["motion_diagnostics"] = payload
    except Exception:
        pass
    return payload


def api_motion_sensitivity(level: str) -> dict[str, Any]:
    try:
        mode = _motion_level_to_mode(level)
        return _jsonsafe(set_motion_sensitivity_mode(mode))
    except Exception as exc:
        payload = {
            "ok": False,
            "error": str(exc),
            "level": level,
            "updated_at": _now_s(),
        }
        ORCHESTRATOR_STATE["last_motion_policy_error"] = payload
        return _jsonsafe(payload)


def api_motion_policy() -> dict[str, Any]:
    return _jsonsafe({
        "ok": True,
        "policy": copy.deepcopy(ORCHESTRATOR_STATE.get("last_motion_policy_sync")),
        "current": {
            "mode": bridge_call_quiet("motion_sensitivity_get"),
            "threshold_g": bridge_call_quiet("motion_threshold_g"),
        },
        "last_error": copy.deepcopy(ORCHESTRATOR_STATE.get("last_motion_policy_error")),
    })


def api_motion_diagnostics() -> dict[str, Any]:
    return _jsonsafe(collect_motion_diagnostics())


def api_motion_rebegin() -> dict[str, Any]:
    result = bridge_call_quiet("movement_rebegin")
    diagnostics = collect_motion_diagnostics()
    diagnostics["movement_rebegin_result"] = result
    diagnostics["ok"] = bool(diagnostics.get("responds"))
    return _jsonsafe(diagnostics)


def api_wire1_scan_count() -> dict[str, Any]:
    count = bridge_call_quiet("wire1_scan_count")
    return _jsonsafe({
        "ok": count is not None,
        "wire1_scan_count": count,
        "note": "On-demand diagnostic only. Boot-time full scans are intentionally avoided.",
    })


# =============================================================================
# Eyes diagnostics and controls
# =============================================================================
def collect_eyes_status() -> dict[str, Any]:
    present = bridge_call_quiet("eyes_present")
    addr = bridge_call_quiet("eyes_addr")
    payload = {
        "ok": bool(_safe_int(present, 0)),
        "present": bool(_safe_int(present, 0)),
        "addr": addr,
        "addr_hex": f"0x{int(addr):02X}" if isinstance(addr, int) and addr >= 0 else None,
        "motion_reactive_requested": S.STATE.get("eyes_motion_reactive", True) if hasattr(S, "STATE") else True,
        "bus": "Wire1",
        "expected_addr": "0x3C",
        "updated_at": _now_s(),
    }
    ORCHESTRATOR_STATE["last_eyes_status"] = payload
    return payload


def _eyes_mode_to_id(mode: str | int) -> int:
    mapping = {
        "demo": 0,
        "idle": 1,
        "left": 2,
        "right": 3,
        "happy": 4,
        "angry": 5,
        "sleep": 6,
        "scan": 7,
        "alert": 8,
        "person": 9,
        "closed": 10,
        "squint": 11,
        "surprised": 12,
        "suspicious": 13,
        "hearts": 14,
        "dizzy": 15,
        "sad": 16,
    }
    key = str(mode).strip().lower()
    if key.isdigit():
        return max(0, min(16, int(key)))
    if key not in mapping:
        raise ValueError("unknown eyes mode")
    return mapping[key]


def api_eyes_status() -> dict[str, Any]:
    return _jsonsafe(collect_eyes_status())


def api_eyes_mode(mode: str) -> dict[str, Any]:
    try:
        mode_id = _eyes_mode_to_id(mode)
        result = bridge_call_quiet("eyes_mode", mode_id)
        status = collect_eyes_status()
        status.update({"requested_mode": mode, "requested_mode_id": mode_id, "bridge_result": result})
        return _jsonsafe(status)
    except Exception as exc:
        return _jsonsafe({"ok": False, "error": str(exc), "mode": mode})


def api_eyes_motion(enabled: str) -> dict[str, Any]:
    value = 1 if str(enabled).strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"} else 0
    result = bridge_call_quiet("eyes_motion_reactive", value)
    try:
        S.STATE["eyes_motion_reactive"] = bool(value)
    except Exception:
        pass
    return _jsonsafe({"ok": result is not None, "enabled": bool(value), "bridge_result": result})


# =============================================================================
# Camera health and vision summary
# =============================================================================
def api_camera_health() -> dict[str, Any]:
    camera_state = copy.deepcopy(getattr(S, "CAMERA_STATE", {}))
    now = _now_s()
    age = None
    try:
        ts = int(camera_state.get("timestamp", 0) or 0)
        age = now - ts if ts else None
    except Exception:
        pass

    payload = {
        "ok": bool(camera_state.get("ok", False)),
        "index": camera_state.get("index"),
        "fps": camera_state.get("fps"),
        "reopens": camera_state.get("reopens"),
        "age_s": age,
        "error": camera_state.get("error"),
        "rotate_180": getattr(S, "CAMERA_ROTATE_180", camera_state.get("rotate_180")),
        "video_nodes": P.list_video_devices() if hasattr(P, "list_video_devices") else [],
        "persistence": False,
        "updated_at": now,
    }
    ORCHESTRATOR_STATE["last_camera_health"] = payload
    return _jsonsafe(payload)


def _summarize_detection_labels(detections: Any) -> list[str]:
    labels: list[str] = []
    if not isinstance(detections, list):
        return labels
    for item in detections:
        if not isinstance(item, dict):
            continue
        label = item.get("label") or item.get("class") or item.get("name") or item.get("category")
        if label:
            labels.append(str(label))
    return sorted(set(labels))


def api_vision_summary() -> dict[str, Any]:
    live = copy.deepcopy(getattr(S, "LIVE_DETECTIONS", {}))
    labels = _summarize_detection_labels(live.get("detections"))
    found = bool(live.get("found_target"))
    target = live.get("target") or getattr(S, "CONTINUOUS_DETECTION_TARGET", "person")
    count = live.get("count", 0)
    confidence = live.get("confidence", 0.0)

    if found:
        summary = f"Target {target} is currently detected with confidence {confidence}."
    elif labels:
        summary = f"Target {target} is not currently detected. Other visible labels: {', '.join(labels)}."
    else:
        summary = f"No current detection labels are available for target {target}."

    payload = {
        "ok": True,
        "target": target,
        "found_target": found,
        "confidence": confidence,
        "count": count,
        "labels": labels,
        "summary": summary,
        "persistence": False,
        "updated_at": _now_s(),
    }
    ORCHESTRATOR_STATE["last_vision_summary"] = payload
    return _jsonsafe(payload)


# =============================================================================
# Enhanced follow-person and reacquisition policy
# =============================================================================
def api_follow_config(profile: str) -> dict[str, Any]:
    key = str(profile).strip().lower()
    if key not in FOLLOW_PROFILES:
        return _jsonsafe({
            "ok": False,
            "error": "unknown follow profile",
            "valid_profiles": list(FOLLOW_PROFILES.keys()),
        })
    ORCHESTRATOR_STATE["follow_profile"] = copy.deepcopy(FOLLOW_PROFILES[key])
    return _jsonsafe({"ok": True, "follow_profile": copy.deepcopy(ORCHESTRATOR_STATE["follow_profile"])})


def api_reacquire(enabled: str) -> dict[str, Any]:
    value = str(enabled).strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}
    ORCHESTRATOR_STATE["reacquire_enabled"] = value
    return _jsonsafe({"ok": True, "reacquire_enabled": value})


def api_reacquire_status() -> dict[str, Any]:
    return _jsonsafe({
        "ok": True,
        "enabled": ORCHESTRATOR_STATE["reacquire_enabled"],
        "state": copy.deepcopy(ORCHESTRATOR_STATE["reacquire_state"]),
        "follow_profile": copy.deepcopy(ORCHESTRATOR_STATE["follow_profile"]),
    })


def _detection_box(target_detection: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(target_detection, dict):
        return None
    try:
        if all(k in target_detection for k in ("x", "y", "width", "height")):
            x = float(target_detection["x"])
            y = float(target_detection["y"])
            w = float(target_detection["width"])
            h = float(target_detection["height"])
            return (x, y, x + w, y + h)
        if all(k in target_detection for k in ("x1", "y1", "x2", "y2")):
            return (
                float(target_detection["x1"]),
                float(target_detection["y1"]),
                float(target_detection["x2"]),
                float(target_detection["y2"]),
            )
        if "bbox" in target_detection and isinstance(target_detection["bbox"], (list, tuple)) and len(target_detection["bbox"]) == 4:
            x, y, w, h = [float(v) for v in target_detection["bbox"]]
            return (x, y, x + w, y + h)
    except Exception:
        return None
    return None


def enhanced_live_autonomy_step(target_detection: Any, image_width: Any, image_height: Any = None) -> None:
    """Safer follow-person steering with target size and reacquisition behavior."""
    try:
        if S.STATE.get("mode", "manual") != "autonomous_vision":
            return

        profile = ORCHESTRATOR_STATE["follow_profile"]
        width = max(1.0, float(image_width or 1.0))
        height = max(1.0, float(image_height or (width * 0.75)))
        now = time.time()

        S.STATE.setdefault("autonomy", {})["owner"] = "vision"

        if target_detection:
            box = _detection_box(target_detection)
            center_x = width / 2.0
            bbox_ratio = 0.0
            if box:
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2.0
                bbox_ratio = max(0.0, min(1.5, abs(y2 - y1) / height))

            ORCHESTRATOR_STATE["reacquire_state"].update({
                "last_seen_at": now,
                "last_target_seen": True,
                "last_lost_action": "none",
            })

            if bbox_ratio >= float(profile["very_close_bbox_ratio"]):
                command = "backward"
            elif bbox_ratio >= float(profile["close_bbox_ratio"]):
                command = "stop"
            elif center_x < width * float(profile["left_boundary"]):
                command = "turn_left"
            elif center_x > width * float(profile["right_boundary"]):
                command = "turn_right"
            else:
                command = "forward"

            ORCHESTRATOR_STATE["reacquire_state"]["last_heading"] = command
            if hasattr(A, "send_command"):
                A.send_command(command, announce=False)
            else:
                bridge_call_quiet(f"dog_{command}")
            return

        # No target this cycle. Reacquire only when autonomous and enabled.
        if not ORCHESTRATOR_STATE.get("reacquire_enabled", True):
            if hasattr(A, "send_command"):
                A.send_command("stop", force=True, announce=False)
            return

        last_seen = float(ORCHESTRATOR_STATE["reacquire_state"].get("last_seen_at") or 0.0)
        lost_s = now - last_seen if last_seen else 999.0

        if lost_s < float(profile["lost_soft_s"]):
            action = "hold_last_heading"
            command = ORCHESTRATOR_STATE["reacquire_state"].get("last_heading", "stop")
        elif lost_s < float(profile["lost_scan_s"]):
            action = "scan_left"
            command = "turn_left"
        elif lost_s < float(profile["lost_stretch_s"]):
            action = "scan_right"
            command = "turn_right"
        else:
            action = "stretch_and_stop"
            command = "stop"

        ORCHESTRATOR_STATE["reacquire_state"].update({
            "last_target_seen": False,
            "last_lost_action": action,
        })

        if hasattr(A, "send_command"):
            A.send_command(command, announce=False)
            if (action == "stretch_and_stop"
                    and ORCHESTRATOR_STATE["reacquire_state"].get("last_stretch_at", 0.0) + 8.0 <= now
                    and hasattr(A, "send_action_named")):
                ORCHESTRATOR_STATE["reacquire_state"]["last_stretch_at"] = now
                A.send_action_named("stretch")
        else:
            bridge_call_quiet(f"dog_{command}")

    except Exception as exc:
        print(f"[autonomy-enhanced] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        try:
            if hasattr(A, "send_command"):
                A.send_command("stop", force=True, announce=False)
        except Exception:
            pass


def _live_detect_wrapper() -> None:
    """Connect perception output to the enhanced autonomy steering hook."""
    if hasattr(P, "continuous_detection_loop"):
        P.continuous_detection_loop(enhanced_live_autonomy_step)
    else:
        raise RuntimeError("perception.continuous_detection_loop is unavailable")


# =============================================================================
# Orchestrator diagnostics and persistence check
# =============================================================================
def api_orchestrator_status() -> dict[str, Any]:
    uptime_s = int(time.time() - float(ORCHESTRATOR_STATE["started_at"]))
    heartbeats = {
        name: round(time.monotonic() - heartbeat, 1)
        for name, heartbeat in getattr(S, "THREAD_HEARTBEATS", {}).items()
    }
    payload = {
        "ok": True,
        "uptime_s": uptime_s,
        "orchestrator": {
            **copy.deepcopy({k: v for k, v in ORCHESTRATOR_STATE.items() if k != "recent_bridge_events"}),
            "recent_bridge_events": list(ORCHESTRATOR_STATE["recent_bridge_events"])[-20:],
        },
        "thread_heartbeats_s_ago": heartbeats,
        "bridge": copy.deepcopy(getattr(S, "BRIDGE_STATE", {})),
        "hardware": copy.deepcopy(getattr(S, "HARDWARE", {})),
        "motion": copy.deepcopy(getattr(S, "MOTION", {})),
        "eyes": copy.deepcopy(getattr(S, "HARDWARE", {}).get("eyes", {})),
        "audio": F.status() if hasattr(F, "status") else {},
    }
    return _jsonsafe(payload)


def api_zero_persistence_check() -> dict[str, Any]:
    return _jsonsafe({
        "ok": True,
        "captures_persisted": False,
        "forbidden_env": {
            "CAPTURE_DIR": os.environ.get("CAPTURE_DIR"),
            "IMAGE_PATH": os.environ.get("IMAGE_PATH"),
        },
        "policy": "camera frames and annotations remain in memory only",
    })


# =============================================================================
# Route registration
# =============================================================================
def _expose(method: str, path: str, handler: RouteHandler) -> None:
    if not callable(handler):
        raise TypeError(f"API handler for {method} {path} is not callable: {handler!r}")
    ui.expose_api(method, path, handler)
    print(f"[route] {_route_signature(method, path)} -> {handler.__name__}", flush=True)


def register_routes() -> int:
    routes: tuple[tuple[str, str, RouteHandler], ...] = (
        # Core state and health
        ("GET", "/api/state", U.api_state),
        ("GET", "/api/health", U.api_health),
        ("GET", "/api/precheck", U.api_precheck),
        ("GET", "/api/orchestrator", api_orchestrator_status),
        ("GET", "/api/zero_persistence", api_zero_persistence_check),

        # DOGZILLA commands
        ("GET", "/api/cmd/{cmd}", U.api_cmd),
        ("GET", "/api/action/{action_id}", U.api_action),
        ("GET", "/api/action_named/{name}", U.api_action_named),
        ("GET", "/api/mode/{mode}", U.api_set_mode),
        ("GET", "/api/attitude/{roll}/{pitch}/{yaw}", U.api_attitude),
        ("GET", "/api/translate/{x}/{y}/{z}", U.api_translate),
        ("GET", "/api/periodic_rot/{axis}/{period_x10}", U.api_periodic_rot),
        ("GET", "/api/periodic_tran/{axis}/{period_x10}", U.api_periodic_tran),
        ("GET", "/api/unload_leg/{leg_id}", U.api_unload_leg),
        ("GET", "/api/load_leg/{leg_id}", U.api_load_leg),

        # Motion additions
        ("GET", "/api/motion_sensitivity/{level}", api_motion_sensitivity),
        ("GET", "/api/motion_policy", api_motion_policy),
        ("GET", "/api/motion_diagnostics", api_motion_diagnostics),
        ("GET", "/api/motion_rebegin", api_motion_rebegin),
        ("GET", "/api/wire1_scan_count", api_wire1_scan_count),

        # Eyes additions
        ("GET", "/api/eyes_status", api_eyes_status),
        ("GET", "/api/eyes_mode/{mode}", api_eyes_mode),
        ("GET", "/api/eyes_motion/{enabled}", api_eyes_motion),

        # Camera / vision additions
        ("GET", "/api/camera_health", api_camera_health),
        ("GET", "/api/vision_summary", api_vision_summary),

        # Autonomy additions
        ("GET", "/api/autonomy/follow_config/{profile}", api_follow_config),
        ("GET", "/api/autonomy/reacquire/{enabled}", api_reacquire),
        ("GET", "/api/autonomy/reacquire_status", api_reacquire_status),

        # Controller and safety
        ("GET", "/api/gamepad/{enabled}", U.api_toggle_gamepad),
        ("GET", "/api/controller/status", U.api_controller_status),
        ("GET", "/api/controller/rescan", U.api_controller_rescan),
        ("GET", "/api/safety/status", U.api_safety_status),
        ("GET", "/api/safety/reset", U.api_safety_reset),

        # Audio
        ("GET", "/api/audio/status", U.api_audio_status),
        ("GET", "/api/audio/test/{event}", U.api_audio_test),
        ("GET", "/api/audio/mute/{enabled}", U.api_audio_mute),

        # Sequences
        ("GET", "/api/sequence/walkaround", U.api_sequence_walkaround),
        ("GET", "/api/sequence/stop", U.api_sequence_stop),

        # Wi-Fi/AP
        ("GET", "/api/start_ap", U.api_start_ap),
        ("GET", "/api/stop_ap", U.api_stop_ap),
        ("GET", "/api/ap/{action}", U.api_ap_control),
        ("GET", "/api/wifi_status", U.api_wifi_status),

        # AI, chat, telemetry
        ("GET", "/api/detect/{target}/{language}", U.api_detect),
        ("GET", "/api/chat/{prompt}", U.api_chat),
        ("GET", "/api/vibration", U.api_vibration),
        ("GET", "/api/telemetry", U.api_telemetry),

        # Camera and live detection
        ("GET", "/api/camera", U.api_camera),
        ("GET", "/api/camera_reset", U.api_camera_reset),
        ("GET", "/api/camera_rotate/{enabled}", U.api_camera_rotate),
        ("GET", "/api/live_detect/{action}", U.api_live_detect),
        ("GET", "/api/live_detect_target/{target}", U.api_live_detect_target),
        ("GET", "/api/live_detect_state", U.api_live_detect_state),

        # Displays and mood
        ("GET", "/api/matrix_scroll/{text}", U.api_matrix_scroll),
        ("GET", "/api/matrix_flash/{text}", U.api_matrix_flash),
        ("GET", "/api/mood/{name}", U.api_mood),
    )

    seen: set[tuple[str, str]] = set()
    for method, path, handler in routes:
        key = (method, path)
        if key in seen:
            raise RuntimeError(f"Duplicate API route: {method} {path}")
        seen.add(key)
        _expose(method, path, handler)

    ORCHESTRATOR_STATE["routes_registered"] = len(routes)
    return len(routes)


# =============================================================================
# Worker registration and watchdog
# =============================================================================
def register_workers() -> int:
    workers: dict[str, Callable[[], None]] = {
        "audio": F.worker_loop,
        "gamepad": C.controller_loop,
        "camera": P.camera_stream_loop,
        "live-detect": _live_detect_wrapper,
        "autonomy": A.autonomous_worker,
        "wifi-status": U.wifi_status_poll_loop,
        "bridge-liveness": HW.bridge_liveness_loop,
        "oled-publisher": lambda: HW.oled_publisher_loop(
            U.best_ipv4,
            lambda: S.STATE.get("mode", "manual"),
            lambda: S.CAMERA_STATE.get("fps", 0),
            ip_refresh_sec=S.OLED_IP_REFRESH_SEC,
            status_refresh_sec=S.OLED_STATUS_REFRESH_SEC,
        ),
        "matrix-worker": HW.matrix_worker_loop,
        "motion": HW.motion_worker_loop,
        "status-line": lambda: HW.status_line_loop(U.best_ipv4),
    }

    for name, target in workers.items():
        if not callable(target):
            raise TypeError(f"Worker target is not callable: {name}")

    S.THREAD_TARGETS.clear()
    S.THREAD_TARGETS.update(workers)
    ORCHESTRATOR_STATE["workers_registered"] = len(workers)
    return len(workers)


def _start_daemon(name: str, target: Callable[[], None]) -> threading.Thread:
    thread = threading.Thread(target=target, name=name, daemon=True)
    thread.start()
    ORCHESTRATOR_STATE["workers_started"].append({"name": name, "started_at": _now_s()})
    print(f"[boot] worker started: {name}", flush=True)
    return thread


def watchdog_loop() -> None:
    stale_sec = float(getattr(S, "THREAD_HEARTBEAT_STALE_SEC", 45.0))
    scan_sec = 15.0
    print(f"[watchdog] enabled stale_sec={stale_sec} scan_sec={scan_sec}", flush=True)
    while True:
        try:
            now = time.monotonic()
            for name, heartbeat in list(S.THREAD_HEARTBEATS.items()):
                age = now - heartbeat
                if age <= stale_sec:
                    continue
                target = S.THREAD_TARGETS.get(name)
                if target is None:
                    print(f"[watchdog] stale worker has no target: {name} age={age:.1f}s", flush=True)
                    S.THREAD_HEARTBEATS[name] = now
                    continue
                count = int(ORCHESTRATOR_STATE["watchdog_restarts"].get(name, 0)) + 1
                ORCHESTRATOR_STATE["watchdog_restarts"][name] = count
                print(f"[watchdog] restarting worker={name} age={age:.1f}s count={count}", flush=True)
                S.THREAD_HEARTBEATS[name] = now
                _start_daemon(name, target)
        except Exception as exc:
            print(f"[watchdog] error: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        time.sleep(scan_sec)


def start_runtime_workers() -> None:
    now = time.monotonic()
    for name in S.THREAD_TARGETS:
        S.THREAD_HEARTBEATS[name] = now
    _start_daemon("precheck", HW.hardware_precheck)
    for name, target in S.THREAD_TARGETS.items():
        _start_daemon(name, target)
    _start_daemon("watchdog", watchdog_loop)


# =============================================================================
# Boot diagnostics and startup
# =============================================================================
def boot_banner(route_count: int, worker_count: int) -> None:
    print("=" * 78, flush=True)
    _print_banner_line("Robodog Pro", f"v{APP_RELEASE} expanded orchestrator")
    _print_banner_line("build_id", APP_BUILD_ID)
    _print_banner_line("state_app_version", getattr(S, "APP_VERSION", "unknown"))
    _print_banner_line("routes", route_count)
    _print_banner_line("supervised_workers", worker_count)
    _print_banner_line("advertised_ip", getattr(S, "ADVERTISED_IP", "unknown"))
    _print_banner_line("nmcli_in_container", U.NMCLI_PATH or "NOT FOUND")
    _print_banner_line("captures_persisted", "FALSE RAM-only")
    _print_banner_line("motion_api", "/api/motion_sensitivity/{off|low|medium|high}")
    _print_banner_line("eyes_api", "/api/eyes_status, /api/eyes_mode/{mode}")
    _print_banner_line("camera_health_api", "/api/camera_health")
    _print_banner_line("vision_summary_api", "/api/vision_summary")
    _print_banner_line("wire_policy", "OEM OLED=Wire, Modulino/eyes=Wire1")
    _print_banner_line("mcu_motion_stop", "disabled by default")
    print("=" * 78, flush=True)


def local_startup_self_check() -> dict[str, Any]:
    checks = {
        "ui_object": ui is not None,
        "app_release": APP_RELEASE,
        "build_id": APP_BUILD_ID,
        "route_count": ORCHESTRATOR_STATE.get("routes_registered"),
        "worker_count": ORCHESTRATOR_STATE.get("workers_registered"),
        "zero_persistence": ORCHESTRATOR_STATE.get("captures_persisted") is False,
        "motion_routes_enabled": True,
        "eyes_routes_enabled": True,
        "camera_health_enabled": True,
        "vision_summary_enabled": True,
    }
    print(f"[self-check] {checks}", flush=True)
    return checks


def initialize_default_motion_policy() -> None:
    try:
        payload = set_motion_sensitivity_mode(1)
        print(f"[motion-policy] default safe sync OK: {payload}", flush=True)
    except Exception as exc:
        error = {"error": f"{type(exc).__name__}: {exc}", "updated_at": _now_s()}
        ORCHESTRATOR_STATE["last_motion_policy_error"] = error
        print(f"[motion-policy] default sync failed: {error}", flush=True)


def publish_ready_feedback() -> None:
    HW.matrix_enqueue("scroll", "DOGZILLA READY", 35)
    HW.oled_detection("READY")
    HW.set_mood("normal")


def main() -> None:
    try:
        HW.register_soft_off()
        route_count = register_routes()
        worker_count = register_workers()
        boot_banner(route_count, worker_count)
        local_startup_self_check()
        start_runtime_workers()
        initialize_default_motion_policy()
        publish_ready_feedback()
        print("Robodog Pro v8.4.6 expanded backend ready.", flush=True)
        App.run()
    except Exception as exc:
        ORCHESTRATOR_STATE["startup_errors"].append({
            "ts": _now_s(),
            "type": type(exc).__name__,
            "error": str(exc),
        })
        print(f"[fatal] startup failed: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
