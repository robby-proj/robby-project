"""
ui_api.py - Robodog Pro v8.4.6 WebUI API and network-status boundary.

This file is complete and intended to replace python/ui_api.py.
It preserves the v8.4 API surface and adds v8.4.6 companion handlers for:
    GET /api/motion_sensitivity/{off|low|medium|high|0|1|2|3}
    GET /api/motion_policy
    GET /api/motion_diagnostics
    GET /api/motion_rebegin
    GET /api/wire1_scan_count
    GET /api/eyes_status
    GET /api/eyes_mode/{mode}
    GET /api/eyes_motion/{enabled}
    GET /api/camera_health
    GET /api/vision_summary
    GET /api/zero_persistence

Design goals:
- Keep camera frames in memory only
- Never pretend the container can perform privileged NetworkManager AP changes
- Keep controller safety latches owned by input_controller.py
- Add motion sensitivity without allowing MCU motion noise to randomly STOP DOGZILLA
- Add Motion, eyes, camera, and vision diagnostics used by the expanded v8.4.6 main.py
"""

from __future__ import annotations

import copy
import fcntl
import ipaddress
import os
import shutil
import socket
import struct
import subprocess
import threading
import time
from urllib.parse import unquote

import autonomy as A
import audio_feedback as F
import hw_bridge as HW
import input_controller as C
import perception as P
import state as S

_TRUE = {"1", "true", "yes", "on", "enable", "enabled"}
_FALSE = {"0", "false", "no", "off", "disable", "disabled"}
_PRECHECK_LOCK = threading.Lock()
_NMCLI_CANDIDATES = ["/usr/bin/nmcli", "/usr/sbin/nmcli", "/bin/nmcli", "/sbin/nmcli"]


def _text(value, name="value", maximum=120):
    result = unquote(str(value)).strip()
    if not result:
        raise ValueError(f"{name} is empty")
    if len(result) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    if any(ord(ch) < 32 and ch not in "\t" for ch in result):
        raise ValueError(f"{name} contains control characters")
    return result


def _bool(value, name="value"):
    result = _text(value, name, 16).lower()
    if result in _TRUE:
        return True
    if result in _FALSE:
        return False
    raise ValueError(f"{name} must be true/false, on/off, yes/no, or 1/0")


def _int(value, name="value", minimum=None, maximum=None):
    try:
        result = int(_text(value, name, 16))
    except Exception as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _snapshot(value):
    return copy.deepcopy(value)


def _ok(**values):
    return HW.jsonsafe({"ok": True, **values})


def _error(error, code="bad_request", **values):
    return HW.jsonsafe({"ok": False, "error": str(error), "code": code, **values})


def _bridge_call(name, *args, quiet=False):
    if quiet and hasattr(HW, "bridge_call_quiet"):
        return HW.bridge_call_quiet(name, *args)
    return HW.bridge_call(name, *args)


def _find_nmcli():
    path = shutil.which("nmcli")
    if path:
        return path
    for candidate in _NMCLI_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


NMCLI_PATH = _find_nmcli()
try:
    S.STATE.setdefault("wifi_status", {})["nmcli_available"] = NMCLI_PATH is not None
except Exception:
    pass


def _read_usb_c_mode():
    try:
        with open(S.USB_MODE_FILE) as file:
            return file.read().strip() or "unknown"
    except Exception:
        return "unknown"


def _disk_free_mb(path="/"):
    try:
        st = os.statvfs(path)
        return int((st.f_bavail * st.f_frsize) / (1024 * 1024))
    except Exception:
        return -1


def _iface_ipv4(iface):
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        packed = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", iface.encode()[:15]))
        return socket.inet_ntoa(packed[20:24])
    except Exception:
        return None
    finally:
        try:
            if sock:
                sock.close()
        except Exception:
            pass


def _is_reachable_ip(ip):
    try:
        addr = ipaddress.ip_address(str(ip))
    except ValueError:
        return False
    return bool(
        addr.version == 4
        and not addr.is_loopback
        and not addr.is_link_local
        and not addr.is_unspecified
        and addr not in ipaddress.ip_network("172.16.0.0/12")
    )


def best_ipv4():
    candidates = []
    for key in ("HOST_IP", "APP_HOST", "ADVERTISED_IP"):
        value = os.environ.get(key)
        if value:
            candidates.append(value)
    for iface in (getattr(S, "AP_INTERFACE", "wlan0"), "wlan0", "end0", "eth0"):
        value = _iface_ipv4(iface)
        if value:
            candidates.append(value)
    for candidate in candidates:
        if _is_reachable_ip(candidate):
            try:
                S.ADVERTISED_IP = candidate
            except Exception:
                pass
            return candidate
    return getattr(S, "ADVERTISED_IP", "0.0.0.0")


def _arp_client_count(iface):
    try:
        count = 0
        with open("/proc/net/arp") as file:
            next(file)
            for line in file:
                parts = line.split()
                if len(parts) >= 6 and parts[5] == iface and parts[2] in ("0x2", "0x6"):
                    count += 1
        return count
    except Exception:
        return 0


def _nmcli_active_wifi():
    if not NMCLI_PATH:
        return (None, None)
    try:
        result = subprocess.run(
            [NMCLI_PATH, "-t", "-f", "NAME,TYPE,DEVICE", "connection", "show", "--active"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        profile = None
        for line in result.stdout.splitlines():
            fields = line.split(":")
            if len(fields) >= 3 and fields[1] == "802-11-wireless" and fields[2] == getattr(S, "AP_INTERFACE", "wlan0"):
                profile = fields[0]
                break
        if not profile:
            return (None, None)
        result2 = subprocess.run(
            [NMCLI_PATH, "-t", "-f", "802-11-wireless.ssid", "connection", "show", profile],
            capture_output=True,
            text=True,
            timeout=3,
        )
        ssid = None
        for line in result2.stdout.splitlines():
            if line.startswith("802-11-wireless.ssid:"):
                ssid = line.split(":", 1)[1].strip()
                break
        return (profile, ssid)
    except Exception:
        return (None, None)


def get_wifi_status():
    iface = getattr(S, "AP_INTERFACE", "wlan0")
    ip = _iface_ipv4(iface)
    profile, ssid = _nmcli_active_wifi()
    ap_ip = getattr(S, "AP_STATIC_IP", "10.42.0.1")
    ap_profile = getattr(S, "AP_PROFILE_NAME", "dogzilla-ap")
    ap_ssid = getattr(S, "AP_SSID", "DOGZILLA-UNOQ")
    if not ip:
        mode = "disconnected"
    elif ip == ap_ip or profile == ap_profile:
        mode = "ap"
        ssid = ssid or ap_ssid
        profile = profile or ap_profile
    else:
        mode = "client"
    return {
        "ok": True,
        "mode": mode,
        "ssid": ssid if ssid else (ap_ssid if mode == "ap" else None),
        "ip": ip,
        "profile": profile,
        "client_count": _arp_client_count(iface) if mode == "ap" else 0,
        "nmcli_available": NMCLI_PATH is not None,
        "updated": int(time.time()),
    }


def wifi_status_poll_loop():
    while True:
        try:
            status = get_wifi_status()
            with S.state_lock:
                S.STATE.setdefault("wifi_status", {}).update(status)
            if hasattr(HW, "publish_network_presence"):
                HW.publish_network_presence(best_ipv4(), int(status.get("client_count", 0) or 0))
        except Exception as exc:
            with S.state_lock:
                S.STATE.setdefault("wifi_status", {}).update({
                    "ok": False,
                    "error": str(exc),
                    "nmcli_available": NMCLI_PATH is not None,
                    "updated": int(time.time()),
                })
        try:
            S.heartbeat("wifi-status")
        except Exception:
            pass
        time.sleep(5)


def _host_ap_command(action):
    script = getattr(
        S,
        "AP_HOST_SCRIPT",
        "$HOME/ArduinoApps/robodog-pro-v8-4-6/host/dogzilla-ap.sh",
    )
    return f"{script} {action}"


def api_state():
    try:
        with S.state_lock:
            S.STATE.setdefault("continuous_detection", {}).update(
                enabled=bool(getattr(S, "CONTINUOUS_DETECTION_ENABLED", False)),
                target=getattr(S, "LIVE_DETECTIONS", {}).get("target", getattr(S, "CONTINUOUS_DETECTION_TARGET", "person")),
                interval_sec=getattr(S, "CONTINUOUS_DETECTION_INTERVAL", 0),
            )
            S.STATE["live_detections"] = {
                key: _snapshot(value)
                for key, value in getattr(S, "LIVE_DETECTIONS", {}).items()
                if not str(key).startswith("_")
            }
            S.STATE["mood"] = _snapshot(getattr(S, "MOOD", {}))
            S.STATE["advertised_ip"] = best_ipv4()
            return HW.jsonsafe(_snapshot(S.STATE))
    except Exception as exc:
        return _error(exc, "state_snapshot_failed")


def api_health():
    now = int(time.time())
    now_m = time.monotonic()
    camera_state = getattr(S, "CAMERA_STATE", {})
    bridge_state = getattr(S, "BRIDGE_STATE", {})
    live = getattr(S, "LIVE_DETECTIONS", {})
    heartbeats = {name: round(now_m - ts, 1) for name, ts in getattr(S, "THREAD_HEARTBEATS", {}).items()}
    return HW.jsonsafe({
        "ok": True,
        "app_version": getattr(S, "APP_VERSION", "unknown"),
        "uptime_s": int(time.time() - getattr(S, "START_TIME", time.time())),
        "advertised_ip": best_ipv4(),
        "hardware": _snapshot(getattr(S, "HARDWARE", {})),
        "camera": {
            "ok": camera_state.get("ok"),
            "index": camera_state.get("index"),
            "fps": camera_state.get("fps"),
            "reopens": camera_state.get("reopens"),
            "age_s": now - int(camera_state.get("timestamp", 0) or 0) if camera_state.get("timestamp") else None,
            "error": camera_state.get("error"),
            "video_nodes": P.list_video_devices() if hasattr(P, "list_video_devices") else [],
        },
        "controller": C.status() if hasattr(C, "status") else {},
        "autonomy": S.STATE.get("autonomy"),
        "sequence": S.STATE.get("sequence"),
        "live_detection": {
            "enabled": getattr(S, "CONTINUOUS_DETECTION_ENABLED", False),
            "target": live.get("target"),
            "count": live.get("count"),
            "found_target": live.get("found_target"),
            "confidence": live.get("confidence"),
            "age_s": now - int(live.get("timestamp", 0) or 0) if live.get("timestamp") else None,
            "error": live.get("error"),
        },
        "bridge": {
            "ok": bridge_state.get("ok"),
            "consecutive_errors": bridge_state.get("consecutive_errors"),
            "age_s": now - int(bridge_state.get("last_ok", 0) or 0) if bridge_state.get("last_ok") else None,
            "last_error": bridge_state.get("last_error"),
        },
        "motion": _snapshot(getattr(S, "MOTION", {})),
        "mood": _snapshot(getattr(S, "MOOD", {})),
        "eyes": _snapshot(getattr(S, "HARDWARE", {}).get("eyes", {})),
        "audio": F.status() if hasattr(F, "status") else {},
        "usb_c": {"mode": _read_usb_c_mode()},
        "wifi": _snapshot(S.STATE.get("wifi_status", {})),
        "wifi_fallback": _snapshot(S.STATE.get("wifi_fallback", {})),
        "gamepad": S.STATE.get("gamepad"),
        "disk_free_mb": _disk_free_mb("/"),
        "captures_persisted": False,
        "nmcli_available": NMCLI_PATH is not None,
        "thread_heartbeats_s_ago": heartbeats,
        "matrix_queue_depth": S.MATRIX_QUEUE.qsize() if hasattr(S, "MATRIX_QUEUE") else None,
    })


def _precheck_job():
    if not _PRECHECK_LOCK.acquire(blocking=False):
        return
    try:
        if hasattr(S, "HARDWARE"):
            S.HARDWARE["probed"] = False
        HW.hardware_precheck()
    finally:
        _PRECHECK_LOCK.release()


def api_precheck():
    if _PRECHECK_LOCK.locked():
        return _ok(started=False, running=True, hardware=_snapshot(getattr(S, "HARDWARE", {})))
    threading.Thread(target=_precheck_job, name="api-precheck", daemon=True).start()
    return _ok(started=True, running=True, hardware=_snapshot(getattr(S, "HARDWARE", {})))


def api_cmd(cmd):
    return A.send_command(unquote(str(cmd)).strip())


def api_action(action_id):
    return A.send_action(unquote(str(action_id)).strip())


def api_action_named(name):
    return A.send_action_named(unquote(str(name)).strip().lower())


def api_set_mode(mode):
    return A.set_mode(unquote(str(mode)).strip())


def api_toggle_gamepad(enabled):
    try:
        requested = _bool(enabled, "enabled")
    except ValueError as exc:
        return _error(exc)
    C.set_enabled(requested)
    if hasattr(S, "robot_say"):
        S.robot_say("Wireless controller enabled; rescanning." if requested else "Wireless controller disabled; robot stopped.")
    return {"ok": True, "gamepad_enabled": requested, "controller": C.status()}


def api_controller_status():
    return {"ok": True, "controller": C.status()}


def api_controller_rescan():
    C.request_rescan()
    return {"ok": True, "rescan_requested": True, "controller": C.status()}


def api_sequence_walkaround():
    return A.start_walkaround_sequence()


def api_sequence_stop():
    return A.stop_walkaround_sequence()


def api_start_ap():
    S.STATE["wifi_ap"] = "host action required"
    return _error(
        "AP switching requires the UNO Q host",
        "host_action_required",
        action="up",
        host_command=_host_ap_command("up"),
        ssid=getattr(S, "AP_SSID", "DOGZILLA-UNOQ"),
        password=getattr(S, "AP_PASSWORD", "dogzilla123"),
        url=getattr(S, "AP_URL", "http://10.42.0.1:7000"),
        wifi_fallback=_snapshot(S.STATE.get("wifi_fallback", {})),
    )

def api_stop_ap():
    S.STATE["wifi_ap"] = "host action required"
    return _error("AP switching requires the UNO Q host", "host_action_required", action="down", host_command=_host_ap_command("down"))


def api_ap_control(action):
    try:
        requested = _text(action, "action", 16).lower()
    except ValueError as exc:
        return _error(exc)
    if requested == "status":
        status = get_wifi_status()
        with S.state_lock:
            S.STATE.setdefault("wifi_status", {}).update(status)
            snapshot = _snapshot(S.STATE["wifi_status"])
        return _ok(action="status", wifi_status=snapshot)
    if requested not in {"up", "down", "restart"}:
        return _error(f"unknown action {requested!r}")
    return _error(
        "AP switching requires the UNO Q host",
        "host_action_required",
        action=requested,
        host_command=_host_ap_command(requested),
        wifi_status=get_wifi_status(),
    )


def api_wifi_status():
    status = get_wifi_status()
    with S.state_lock:
        S.STATE.setdefault("wifi_status", {}).update(status)
        wifi_status = _snapshot(S.STATE["wifi_status"])
        wifi_fallback = _snapshot(S.STATE.get("wifi_fallback", {}))
    return _ok(
        wifi_status=wifi_status,
        wifi_fallback=wifi_fallback,
        **status,
    )

def api_detect(target, language):
    return P.run_object_detection(unquote(str(target)).strip(), unquote(str(language)).strip(), A.one_shot_autonomy_apply)


def api_chat(prompt):
    try:
        message = _text(prompt, "prompt", 240)
    except ValueError as exc:
        return _error(exc)
    with S.state_lock:
        S.STATE.setdefault("chat", []).append({"role": "owner", "text": message})
        S.STATE["chat"] = S.STATE["chat"][-40:]
    HW.matrix_enqueue("scroll", message[:60], 35)
    if hasattr(S, "robot_say"):
        S.robot_say(f"Received: {message}")
    return _ok(answer=f"Received: {message}")


def api_vibration():
    return HW.jsonsafe(getattr(S, "MOTION", {}))


def api_telemetry():
    try:
        battery = _bridge_call("dog_battery", quiet=True)
        if hasattr(A, "apply_battery_mood"):
            A.apply_battery_mood(battery)
        data = {
            "ok": True,
            "x": HW.safe_float(_bridge_call("get_accel_x", quiet=True)) if hasattr(HW, "safe_float") else _bridge_call("get_accel_x", quiet=True),
            "y": HW.safe_float(_bridge_call("get_accel_y", quiet=True)) if hasattr(HW, "safe_float") else _bridge_call("get_accel_y", quiet=True),
            "z": HW.safe_float(_bridge_call("get_accel_z", quiet=True)) if hasattr(HW, "safe_float") else _bridge_call("get_accel_z", quiet=True),
            "vibration": HW.safe_float(_bridge_call("get_vibration", quiet=True)) if hasattr(HW, "safe_float") else _bridge_call("get_vibration", quiet=True),
            "battery": battery,
            "roll": HW.safe_float(_bridge_call("dog_read_roll_f", quiet=True), default=None) if hasattr(HW, "safe_float") else None,
            "pitch": HW.safe_float(_bridge_call("dog_read_pitch_f", quiet=True), default=None) if hasattr(HW, "safe_float") else None,
            "yaw": HW.safe_float(_bridge_call("dog_read_yaw_f", quiet=True), default=None) if hasattr(HW, "safe_float") else None,
            "firmware": HW.clean_str(_bridge_call("dog_firmware_version", quiet=True)) if hasattr(HW, "clean_str") else str(_bridge_call("dog_firmware_version", quiet=True)),
        }
        S.STATE["telemetry"] = HW.jsonsafe(data)
        return S.STATE["telemetry"]
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def api_motion_sensitivity(level):
    mapping = {"off": 0, "low": 1, "medium": 2, "med": 2, "high": 3}
    requested = str(level).strip().lower()
    mode = mapping.get(requested)
    if mode is None:
        try:
            mode = int(requested)
        except Exception:
            return _error("level must be off, low, medium, high, or 0..3", "invalid_motion_sensitivity", level=level)
    mode = max(0, min(3, int(mode)))
    try:
        bridge_result = _bridge_call("motion_sensitivity", mode)
        threshold = _bridge_call("motion_threshold_g", quiet=True)
        local_stop = _bridge_call("motion_local_stop_enable", 0)
        S.STATE["motion_sensitivity"] = mode
        S.STATE["motion_threshold_g"] = threshold
        S.STATE["motion_local_stop_enabled"] = bool(local_stop)
        return _ok(mode=mode, threshold_g=threshold, local_stop_enabled=bool(local_stop), bridge_result=bridge_result)
    except Exception as exc:
        return _error(exc, "motion_sensitivity_failed", mode=mode)


# ============================================================================
# v8.4.5 expanded Motion, eyes, camera, and vision diagnostics
# ============================================================================

def _bridge_float(name, default=None):
    try:
        value = _bridge_call(name, quiet=True)
        if hasattr(HW, "safe_float"):
            return HW.safe_float(value, default=default)
        return float(value)
    except Exception:
        return default


def _bridge_int(name, default=None):
    try:
        value = _bridge_call(name, quiet=True)
        return int(value)
    except Exception:
        return default


def api_motion_policy():
    """Return current MCU Motion sensitivity policy without changing it."""
    return _ok(
        mode=_bridge_int("motion_sensitivity_get", None),
        threshold_g=_bridge_float("motion_threshold_g", None),
        local_stop_enabled=bool(S.STATE.get("motion_local_stop_enabled", False)),
        cached_mode=S.STATE.get("motion_sensitivity"),
        cached_threshold_g=S.STATE.get("motion_threshold_g"),
    )


def api_motion_diagnostics():
    """Collect the Motion values needed to debug false impact/STOP behavior."""
    present = _bridge_int("movement_present", 0)
    responds = _bridge_int("movement_responds", 0)
    payload = {
        "ok": bool(present),
        "present": bool(present),
        "responds": bool(responds),
        "addr": _bridge_int("movement_addr", None),
        "x": _bridge_float("get_accel_x", None),
        "y": _bridge_float("get_accel_y", None),
        "z": _bridge_float("get_accel_z", None),
        "movement_mag": _bridge_float("movement_mag", None),
        "vibration": _bridge_float("get_vibration", None),
        "motion_sensitivity": _bridge_int("motion_sensitivity_get", None),
        "threshold_g": _bridge_float("motion_threshold_g", None),
        "impact_confirm_count": _bridge_int("impact_confirm_count", None),
        "impact_cooldown_remaining_ms": _bridge_int("impact_cooldown_remaining_ms", None),
        "motion_reads_count": _bridge_int("motion_reads_count", None),
        "motion_active_count": _bridge_int("motion_active_count", None),
        "local_stop_enabled": bool(S.STATE.get("motion_local_stop_enabled", False)),
        "updated": int(time.time()),
        "interpretation": "At rest, movement_mag should be near 1.0 and vibration should settle near 0.0.",
    }
    try:
        S.STATE["motion_diagnostics"] = payload
    except Exception:
        pass
    return HW.jsonsafe(payload)


def api_motion_rebegin():
    """Request a non-aggressive Movement re-begin, then return diagnostics."""
    result = _bridge_call("movement_rebegin", quiet=True)
    data = api_motion_diagnostics()
    try:
        data["movement_rebegin_result"] = result
    except Exception:
        pass
    return HW.jsonsafe(data)


def api_wire1_scan_count():
    """Count responding Wire1 devices on demand. This is intentionally not a boot task."""
    count = _bridge_int("wire1_scan_count", None)
    return _ok(wire1_scan_count=count, note="On-demand diagnostic only. Boot scan remains skipped for stability.")


_EYES_MODES = {
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


def _eyes_mode_id(mode):
    text = str(mode).strip().lower()
    if text.isdigit():
        return max(0, min(16, int(text)))
    if text not in _EYES_MODES:
        raise ValueError(f"unknown eyes mode {mode!r}")
    return _EYES_MODES[text]


def api_eyes_status():
    if hasattr(HW, "refresh_eyes_health"):
        health = HW.refresh_eyes_health()
    else:
        health = {
            "present": bool(_bridge_int("eyes_present", 0)),
            "mode": _bridge_int("eyes_mode_get", None),
            "flush_ok_count": _bridge_int("eyes_flush_ok_count", None),
            "flush_error_count": _bridge_int("eyes_flush_error_count", None),
            "bus_fault": bool(_bridge_int("eyes_bus_fault", 0)),
        }
    addr = _bridge_int("eyes_addr", -1)
    return _ok(
        present=bool(health.get("present")),
        addr=addr,
        addr_hex=f"0x{addr:02X}" if isinstance(addr, int) and addr >= 0 else None,
        mode=health.get("mode"),
        flush_ok_count=health.get("flush_ok_count"),
        flush_error_count=health.get("flush_error_count"),
        bus_fault=health.get("bus_fault"),
        bus="Wire1",
        bus_hz=100000,
        expected_addr="0x3C",
        motion_reactive=S.STATE.get("eyes_motion_reactive", True),
        recovery_policy="retry, re-probe, reinitialize, clear framebuffer",
    )

def api_eyes_mode(mode):
    try:
        mode_id = _eyes_mode_id(mode)
        result = _bridge_call("eyes_mode", mode_id, quiet=True)
        status = api_eyes_status()
        status["requested_mode"] = mode
        status["requested_mode_id"] = mode_id
        status["bridge_result"] = result
        return HW.jsonsafe(status)
    except Exception as exc:
        return _error(exc, "eyes_mode_failed", mode=mode)


def api_eyes_motion(enabled):
    try:
        value = _bool(enabled, "enabled")
    except ValueError as exc:
        return _error(exc)
    result = _bridge_call("eyes_motion_reactive", 1 if value else 0, quiet=True)
    S.STATE["eyes_motion_reactive"] = value
    return _ok(enabled=value, bridge_result=result)


def api_camera_health():
    now = int(time.time())
    camera_state = _snapshot(getattr(S, "CAMERA_STATE", {}))
    ts = int(camera_state.get("timestamp", 0) or 0)
    return _ok(
        camera={
            "ok": camera_state.get("ok"),
            "index": camera_state.get("index"),
            "fps": camera_state.get("fps"),
            "reopens": camera_state.get("reopens"),
            "age_s": now - ts if ts else None,
            "error": camera_state.get("error"),
            "rotate_180": getattr(S, "CAMERA_ROTATE_180", camera_state.get("rotate_180")),
            "video_nodes": P.list_video_devices() if hasattr(P, "list_video_devices") else [],
            "persistence": False,
        }
    )


def _labels_from_detections(detections):
    labels = []
    if not isinstance(detections, list):
        return labels
    for item in detections:
        if not isinstance(item, dict):
            continue
        label = item.get("label") or item.get("class") or item.get("name") or item.get("category")
        if label:
            labels.append(str(label))
    return sorted(set(labels))


def api_vision_summary():
    live = _snapshot(getattr(S, "LIVE_DETECTIONS", {}))
    labels = _labels_from_detections(live.get("detections"))
    target = live.get("target") or getattr(S, "CONTINUOUS_DETECTION_TARGET", "person")
    found = bool(live.get("found_target"))
    confidence = live.get("confidence", 0.0)
    if found:
        summary = f"Target {target} is currently detected with confidence {confidence}."
    elif labels:
        summary = f"Target {target} is not currently detected. Other visible labels: {', '.join(labels)}."
    else:
        summary = f"No current detection labels are available for target {target}."
    return _ok(
        target=target,
        found_target=found,
        confidence=confidence,
        count=live.get("count", 0),
        labels=labels,
        summary=summary,
        persistence=False,
        updated=int(time.time()),
    )


def api_zero_persistence():
    return _ok(
        captures_persisted=False,
        forbidden_env={
            "CAPTURE_DIR": os.environ.get("CAPTURE_DIR"),
            "IMAGE_PATH": os.environ.get("IMAGE_PATH"),
        },
        policy="camera frames and annotations remain in memory only",
    )


def api_attitude(roll, pitch, yaw):
    try:
        r = _int(roll, "roll", 0, 255)
        p = _int(pitch, "pitch", 0, 255)
        y = _int(yaw, "yaw", 0, 255)
        _bridge_call("dog_attitude", r, p, y)
        return _ok(roll=r, pitch=p, yaw=y)
    except Exception as exc:
        return _error(exc)


def api_translate(x, y, z):
    try:
        ix = _int(x, "x", 0, 255)
        iy = _int(y, "y", 0, 255)
        iz = _int(z, "z", 0, 255)
        _bridge_call("dog_translate", ix, iy, iz)
        return _ok(x=ix, y=iy, z=iz)
    except Exception as exc:
        return _error(exc)


def api_periodic_rot(axis, period_x10):
    try:
        a = _int(axis, "axis", 0, 2)
        p = _int(period_x10, "period_x10", 0, 255)
        _bridge_call("dog_periodic_rot", a, p)
        return _ok(axis=a, period_x10=p)
    except Exception as exc:
        return _error(exc)


def api_periodic_tran(axis, period_x10):
    try:
        a = _int(axis, "axis", 0, 2)
        p = _int(period_x10, "period_x10", 0, 255)
        _bridge_call("dog_periodic_tran", a, p)
        return _ok(axis=a, period_x10=p)
    except Exception as exc:
        return _error(exc)


def api_unload_leg(leg_id):
    try:
        n = _int(leg_id, "leg_id", 1, 4)
        _bridge_call("dog_unload_leg", n)
        return _ok(leg=n, action="unload")
    except Exception as exc:
        return _error(exc)


def api_load_leg(leg_id):
    try:
        n = _int(leg_id, "leg_id", 1, 4)
        _bridge_call("dog_load_leg", n)
        return _ok(leg=n, action="load")
    except Exception as exc:
        return _error(exc)


def api_camera():
    return HW.jsonsafe(_snapshot(getattr(S, "CAMERA_STATE", {})))


def api_camera_reset():
    P.reset_camera_index()
    return _ok(reset=True)


def api_camera_rotate(enabled):
    try:
        value = _bool(enabled, "enabled")
    except ValueError as exc:
        return _error(exc)
    S.CAMERA_ROTATE_180 = value
    S.STATE["camera_rotate_180"] = value
    return _ok(camera_rotate_180=value)


def api_live_detect(action):
    action = unquote(str(action)).strip().lower()
    if action == "on":
        S.CONTINUOUS_DETECTION_ENABLED = True
        HW.matrix_enqueue("scroll", "LIVE ON", 35)
        if hasattr(S, "robot_say"):
            S.robot_say("Live detection ON.")
    elif action == "off":
        S.CONTINUOUS_DETECTION_ENABLED = False
        HW.matrix_enqueue("scroll", "LIVE OFF", 35)
        if hasattr(S, "robot_say"):
            S.robot_say("Live detection OFF.")
    elif action == "status":
        pass
    else:
        return _error(f"unknown live detection action {action!r}")
    return _ok(enabled=S.CONTINUOUS_DETECTION_ENABLED, state={k: v for k, v in S.LIVE_DETECTIONS.items() if not str(k).startswith("_")})


def api_live_detect_target(target):
    normalizer = getattr(P, "normalize_label", lambda value: str(value).strip().lower())
    label = normalizer(unquote(str(target)))
    S.LIVE_DETECTIONS["target"] = label
    HW.matrix_enqueue("scroll", f"TARGET {label.upper()}", 35)
    if hasattr(S, "robot_say"):
        S.robot_say(f"Live target: {label}")
    return _ok(target=label)


def api_live_detect_state():
    return HW.jsonsafe({k: v for k, v in S.LIVE_DETECTIONS.items() if not str(k).startswith("_")})


def api_matrix_scroll(text):
    try:
        value = _text(text, "text", 80)
    except ValueError as exc:
        return _error(exc)
    HW.matrix_enqueue("scroll", value, 35)
    return _ok(text=value)


def api_matrix_flash(text):
    try:
        value = _text(text, "text", 80)
    except ValueError as exc:
        return _error(exc)
    HW.matrix_enqueue("flash", value)
    return _ok(text=value)


def api_mood(name):
    value = unquote(str(name)).strip().lower()
    allowed = {"normal", "happy", "searching", "detected", "alert", "low_energy", "sleeping"}
    if value not in allowed:
        return _error(f"unknown mood {value!r}")
    HW.set_mood(value)
    return _ok(mood=value)


def api_safety_status():
    controller = C.status()
    return HW.jsonsafe({
        "ok": True,
        "latched": bool(controller.get("safety_blocked") or controller.get("hard_stop_latched")),
        "reason": controller.get("safety_reason"),
        "armed": bool(controller.get("armed", False)),
        "bridge_ok": bool(getattr(S, "BRIDGE_STATE", {}).get("ok")),
        "controller": controller,
        "motion": _snapshot(getattr(S, "MOTION", {})),
    })


def api_safety_reset():
    C.request_rescan()
    return _ok(reset_requested=True, neutral_rearm_required=True, message="Rescan requested; return all controls to neutral to re-arm.", controller=C.status())


def api_audio_status():
    events = sorted(getattr(F, "EVENTS", {}).keys())
    return _ok(
        audio=F.status(),
        events=events,
        buzzer_present=bool(getattr(S, "HARDWARE", {}).get("buzzer_present")),
        mcu_event_ids=_snapshot(getattr(F, "BUZZER_EVENT_IDS", {})),
    )

def api_audio_test(event):
    try:
        requested = _text(event, "event", 40).lower()
    except ValueError as exc:
        return _error(exc)
    emitted = bool(F.emit(requested))
    if not emitted:
        return _error("unknown, muted, throttled, or unavailable audio event", "audio_not_emitted", event=requested, audio=F.status())
    return _ok(event=requested, audio=F.status())


def api_audio_mute(enabled):
    try:
        value = _bool(enabled, "enabled")
    except ValueError as exc:
        return _error(exc)
    return _ok(audio=F.set_muted(value))
