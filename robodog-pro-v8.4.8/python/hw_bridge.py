"""
hw_bridge.py - Robodog Pro v8.4.6 RouterBridge boundary.

Every other module talks to the MCU through the helpers here. Display-setting
calls go through a last-value de-dup cache so we don't flood the Bridge with
identical text. Nothing in this module writes to disk.
"""

import math
import queue
import subprocess
import threading
import time

try:
    from arduino.app_utils import Bridge
except ImportError:
    Bridge = None  # allows off-device syntax testing

from state import (
    BRIDGE_STATE, HARDWARE, MATRIX_QUEUE, MOOD,
    BOOT_PROBE_INTERVAL, BOOT_PROBE_TIMEOUT,
    bridge_lock, heartbeat, robot_say,
)

# ============================================================================
# Bridge core
# ============================================================================
def bridge_call(name, *args, quiet=False):
    """Serialized Bridge call. Updates BRIDGE_STATE for /api/health."""
    with bridge_lock:
        try:
            if Bridge is None:
                raise RuntimeError("Bridge not available (off-device)")
            res = Bridge.call(name, *args) if args else Bridge.call(name)
            BRIDGE_STATE["ok"] = True
            BRIDGE_STATE["last_ok"] = int(time.time())
            BRIDGE_STATE["last_error"] = None
            BRIDGE_STATE["consecutive_errors"] = 0
            HARDWARE["bridge_ok"] = True
            return res
        except Exception as e:
            BRIDGE_STATE["ok"] = False
            BRIDGE_STATE["last_error"] = str(e)
            BRIDGE_STATE["consecutive_errors"] += 1
            HARDWARE["bridge_ok"] = False
            if not quiet:
                print(f"[BRIDGE ERROR] {name}: {e}", flush=True)
            return None


def bridge_call_quiet(name, *args):
    return bridge_call(name, *args, quiet=True)


def safe_int(v, default=None):
    try:
        return int(v)
    except Exception:
        return default


def safe_float(v, default=0.0):
    """float() but NaN/inf (which crash JSON) map to the default."""
    try:
        f = float(v)
    except Exception:
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def jsonsafe(obj):
    """Recursively replace NaN/inf floats with None so stdlib json won't 500."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: jsonsafe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonsafe(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(jsonsafe(v) for v in obj)
    return obj


def clean_str(v):
    """None if value is missing/blank/'?', else str(v)."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s == "?":
        return None
    return s


# ============================================================================
# De-dup cache for display calls
# ============================================================================
_dedupe_cache = {
    "oled_ip": None, "oled_status": None, "oled_detection": None,
    "mood": None, "network_presence": None,
    "person_detected": None,
}
_dedupe_lock = threading.Lock()


def _cached_call(cache_key, bridge_name, value):
    with _dedupe_lock:
        if _dedupe_cache.get(cache_key) == value:
            return False
        _dedupe_cache[cache_key] = value
    bridge_call_quiet(bridge_name, value)
    return True


def oled_ip(v):        _cached_call("oled_ip",        "oled_set_ip",        v)
def oled_status(v):    _cached_call("oled_status",    "oled_set_status",    v)
def oled_detection(v): _cached_call("oled_detection", "oled_set_detection", v)


def set_person_detected(found):
    """Publish a detection edge once; the MCU owns the face, LEDs, matrix and tone."""
    value = 1 if bool(found) else 0
    with _dedupe_lock:
        if _dedupe_cache.get("person_detected") == value:
            return False
        _dedupe_cache["person_detected"] = value
    bridge_call_quiet("set_person_detected", value)
    return True


def publish_network_presence(ip, client_count):
    """
    De-duplicated publisher of network state to the MCU.
    * Always keeps the OLED IP line current.
    * Tells the sketch how many AP clients are connected, so the matrix can
      swap the noisy IP banner for a compact LINK/CLIENT indicator.
    """
    ip_text = str(ip or "no-net")
    count = max(0, int(client_count or 0))
    oled_ip(ip_text)
    key = f"{ip_text}|{count}"
    with _dedupe_lock:
        if _dedupe_cache.get("network_presence") == key:
            return
        _dedupe_cache["network_presence"] = key
    bridge_call_quiet("matrix_set_connected", count)


# ============================================================================
# Mood - single 'vibe' knob (matrix + OLED coordinated), de-duped
# ============================================================================
_MOOD_ALIASES = {
    "normal":     ("DOGZILLA",       "READY"),
    "happy":      ("DOGZILLA HAPPY", "HAPPY"),
    "searching":  ("SEARCHING",      "SEARCHING"),
    "detected":   ("PERSON",         "DETECTED"),
    "alert":      ("ALERT",          "ALERT"),
    "low_energy": ("LOW BATTERY",    "LOW BATTERY"),
    "sleeping":   ("SLEEPING",       "SLEEPING"),
}


def set_mood(name):
    if name not in _MOOD_ALIASES:
        return
    if MOOD.get("current") == name:
        return
    MOOD["current"] = name
    matrix_txt, oled_txt = _MOOD_ALIASES[name]
    if name == "searching":
        matrix_enqueue("searching")
    else:
        matrix_enqueue("scroll", matrix_txt, 35)
        # Preserve an underlying search while a low-battery mood temporarily
        # owns the display. All other non-search moods end the funny/search
        # animation on the external Modulino LED Matrix.
        if name != "low_energy":
            bridge_call_quiet("face_matrix_searching_set", 0)
    oled_detection(oled_txt)


# ============================================================================
# Matrix event queue
# ============================================================================
def matrix_enqueue(kind, *args):
    try:
        MATRIX_QUEUE.put_nowait((kind, args))
    except queue.Full:
        pass


def matrix_worker_loop():
    while True:
        try:
            kind, args = MATRIX_QUEUE.get(timeout=1.0)
        except queue.Empty:
            heartbeat("matrix-worker")
            continue
        try:
            if kind == "flash":
                bridge_call_quiet("matrix_flash", args[0])
            elif kind == "person_scroll":
                bridge_call_quiet("matrix_person_scroll", args[0])
            elif kind == "scroll":
                text = args[0]
                speed = int(args[1]) if len(args) > 1 else 35
                bridge_call_quiet("matrix_scroll", text, speed)
            elif kind == "searching":  bridge_call_quiet("matrix_searching")
            elif kind == "person":     bridge_call_quiet("matrix_person")
            elif kind == "idle":       bridge_call_quiet("matrix_idle")
            elif kind == "hello":      bridge_call_quiet("matrix_hello")
            elif kind == "battery":    bridge_call_quiet("matrix_battery")
            elif kind == "alert":      bridge_call_quiet("matrix_alert")
        except Exception as e:
            print(f"[matrix-worker] {e}", flush=True)
        heartbeat("matrix-worker")


# ============================================================================
# OLED publisher - throttled IP + mode/FPS status line
# ============================================================================
def oled_publisher_loop(get_best_ipv4, get_mode, get_fps,
                        ip_refresh_sec=30.0, status_refresh_sec=10.0):
    last_ip = None
    last_status = None
    last_ip_send = 0
    last_status_send = 0
    while True:
        try:
            now = time.time()
            ip = get_best_ipv4()
            ip_disp = ip if ip else "no-net"
            if ip_disp != last_ip or (now - last_ip_send) > ip_refresh_sec:
                oled_ip(ip_disp)
                last_ip = ip_disp
                last_ip_send = now
            mode_txt = get_mode().upper()
            fps = get_fps() or 0
            status_txt = f"DOG {mode_txt[:8]} F{fps}"
            if status_txt != last_status or (now - last_status_send) > status_refresh_sec:
                oled_status(status_txt)
                last_status = status_txt
                last_status_send = now
        except Exception as e:
            print(f"[oled-publisher] {e}", flush=True)
        heartbeat("oled-publisher")
        time.sleep(2)


# ============================================================================
# Bridge and eyes OLED liveness
# ============================================================================
def refresh_eyes_health():
    """Read fail-soft MCU eye-display diagnostics introduced by sketch v8.4.6."""
    present = safe_int(bridge_call_quiet("eyes_present"), 0)
    mode = safe_int(bridge_call_quiet("eyes_mode_get"), None)
    flush_ok = safe_int(bridge_call_quiet("eyes_flush_ok_count"), None)
    flush_errors = safe_int(bridge_call_quiet("eyes_flush_error_count"), None)
    bus_fault = safe_int(bridge_call_quiet("eyes_bus_fault"), None)

    eyes = HARDWARE.setdefault("eyes", {})
    eyes.update({
        "present": bool(present),
        "mode": mode,
        "flush_ok_count": flush_ok,
        "flush_error_count": flush_errors,
        "bus_fault": bool(bus_fault) if bus_fault is not None else None,
        "updated": int(time.time()),
    })
    return dict(eyes)


def refresh_face_matrix_health():
    """Read external 12x8 Modulino LED Matrix status and current expression."""
    present = safe_int(bridge_call_quiet("face_matrix_present"), 0)
    expression = safe_int(bridge_call_quiet("face_matrix_expression"), None)
    low_battery = safe_int(bridge_call_quiet("face_matrix_battery_low"), None)
    threshold_mg = safe_int(
        bridge_call_quiet("face_matrix_aggressive_threshold_mg"), None
    )
    face = HARDWARE.setdefault("face_matrix", {})
    face.update({
        "present": bool(present),
        "expression": expression,
        "expression_name": {
            0: "happy", 1: "sad", 2: "serious", 3: "funny"
        }.get(expression, "unknown"),
        "low_battery": bool(low_battery) if low_battery is not None else None,
        "aggressive_threshold_mg": threshold_mg,
        "updated": int(time.time()),
    })
    return dict(face)


def bridge_liveness_loop():
    while True:
        bridge_call_quiet("get_vibration")
        refresh_eyes_health()
        refresh_face_matrix_health()
        heartbeat("bridge-liveness")
        time.sleep(30)


# ============================================================================
# Hardware precheck
# ============================================================================
def hardware_precheck():
    print("[precheck] waiting for sketch bridge providers", flush=True)
    deadline = time.monotonic() + BOOT_PROBE_TIMEOUT
    while time.monotonic() < deadline:
        present = bridge_call_quiet("movement_present")
        if present is not None:
            HARDWARE["movement_present"]  = bool(safe_int(present, 0))
            HARDWARE["movement_responds"] = bool(safe_int(bridge_call_quiet("movement_responds"), 0))
            HARDWARE["movement_addr"]     = safe_int(bridge_call_quiet("movement_addr"), 0)
            HARDWARE["dog_battery_probe"] = safe_int(bridge_call_quiet("dog_battery"), -1)
            fw = bridge_call_quiet("dog_firmware_version")
            HARDWARE["dog_firmware"]      = str(fw) if fw not in (None, "?") else None
            HARDWARE["buzzer_present"]    = bool(safe_int(bridge_call_quiet("buzzer_present"), 0))
            refresh_eyes_health()
            refresh_face_matrix_health()
            HARDWARE["probed"] = True
            HARDWARE["probe_error"] = None
            addr = HARDWARE["movement_addr"] or 0
            print(f"[precheck] movement_present={HARDWARE['movement_present']} "
                  f"responds={HARDWARE['movement_responds']} addr=0x{addr:02X} "
                  f"battery={HARDWARE['dog_battery_probe']} "
                  f"fw={HARDWARE['dog_firmware']} "
                  f"buzzer={HARDWARE.get('buzzer_present')} "
                  f"face_matrix={HARDWARE.get('face_matrix')} "
                  f"eyes={HARDWARE.get('eyes')}", flush=True)
            matrix_enqueue("scroll", "READY", 35)
            return
        time.sleep(BOOT_PROBE_INTERVAL)
    HARDWARE["probe_error"] = "bridge providers not seen within timeout"
    print(f"[precheck] {HARDWARE['probe_error']}", flush=True)


# ============================================================================
# Motion worker - reads Modulino and publishes to MOTION dict
# ============================================================================
def motion_worker_loop():
    from state import MOTION
    still_since = time.monotonic()
    while True:
        heartbeat("motion")
        present  = bool(safe_int(bridge_call_quiet("movement_present"), 0))
        responds = bool(safe_int(bridge_call_quiet("movement_responds"), 0))
        addr     = safe_int(bridge_call_quiet("movement_addr"), 0)
        x = safe_float(bridge_call_quiet("get_accel_x"))
        y = safe_float(bridge_call_quiet("get_accel_y"))
        z = safe_float(bridge_call_quiet("get_accel_z"))
        vibration = safe_float(bridge_call_quiet("get_vibration"))
        moving = vibration > 0.08
        now = time.monotonic()
        if (abs(x) + abs(y) + abs(z)) > 0.05:
            MOTION["last_nonzero_ts"] = time.time()
        if moving:
            still_since = now
        still_seconds = max(0.0, now - still_since)
        MOTION.update(ok=(present and responds), present=present, responds=responds,
                      addr=addr or 0, x=x, y=y, z=z, vibration=vibration,
                      moving=moving, still_seconds=round(still_seconds, 1))
        HARDWARE["movement_present"]  = present
        HARDWARE["movement_responds"] = responds
        HARDWARE["movement_addr"]     = addr
        time.sleep(0.20)


# ============================================================================
# Soft-off handler (called from sketch via Bridge.call("soft_off"))
# ============================================================================
def _do_soft_off():
    print("[soft_off] MCU requested shutdown", flush=True)
    robot_say("Soft-off requested. Powering down.")
    set_mood("sleeping")
    try:
        subprocess.run(["systemctl", "stop", "applab-user.service"],
                       check=False, timeout=10)
    except Exception as e:
        print(f"[soft_off] service stop failed: {e}", flush=True)
    time.sleep(1)
    try:
        subprocess.run(["systemctl", "poweroff"], check=False, timeout=10)
    except Exception as e:
        print(f"[soft_off] poweroff failed: {e}", flush=True)


def handle_soft_off(*_a, **_kw):
    threading.Thread(target=_do_soft_off, name="soft-off", daemon=True).start()
    return "ok"


def register_soft_off():
    if Bridge is None:
        return
    try:
        Bridge.provide("soft_off", handle_soft_off)
        print("[bridge] soft_off handler registered", flush=True)
    except Exception as e:
        print(f"[bridge] could not register soft_off: {e}", flush=True)


# ============================================================================
# Human-friendly status line (v7.9)
# ============================================================================
def status_line_loop(get_ip):
    """
    Print ONE concise status line every 10s instead of camera spam. Shows the
    things you actually want while debugging without a camera:
      IP, Modulino present/responds, controller connection, Bridge health,
      battery, and current mode.
    """
    from state import STATE, MOTION
    while True:
        heartbeat("status-line")
        try:
            ctrl = STATE.get("controller", {})
            ctrl_txt = ("connected:" + str(ctrl.get("api"))
                        if ctrl.get("connected") else (ctrl.get("error") or "scanning"))
            line = (
                "[status] "
                f"ip={get_ip()} "
                f"mode={STATE.get('mode')} "
                f"bridge={'OK' if BRIDGE_STATE.get('ok') else 'ERR'} "
                f"modulino={'present' if MOTION.get('present') else 'absent'}"
                f"/{'responds' if MOTION.get('responds') else 'silent'} "
                f"vib={MOTION.get('vibration')} "
                f"controller={ctrl_txt} "
                f"events={ctrl.get('events', 0)} "
                f"battery={HARDWARE.get('dog_battery_probe')} "
                f"fw={HARDWARE.get('dog_firmware')} "
                f"face={HARDWARE.get('face_matrix', {}).get('expression_name')} "
                f"eyes_fault={HARDWARE.get('eyes', {}).get('bus_fault')} "
                f"eyes_err={HARDWARE.get('eyes', {}).get('flush_error_count')}"
            )
            print(line, flush=True)
        except Exception as e:
            print(f"[status] error: {e}", flush=True)
        time.sleep(10)


# ============================================================================
# Watchdog
# ============================================================================
def watchdog_loop():
    from state import THREAD_HEARTBEATS, THREAD_TARGETS, THREAD_HEARTBEAT_STALE_SEC
    while True:
        try:
            now = time.monotonic()
            for name, hb in list(THREAD_HEARTBEATS.items()):
                age = now - hb
                if age > THREAD_HEARTBEAT_STALE_SEC:
                    print(f"[watchdog] '{name}' stale {int(age)}s", flush=True)
                    target = THREAD_TARGETS.get(name)
                    if target is not None:
                        THREAD_HEARTBEATS[name] = now
                        t = threading.Thread(target=target, name=name, daemon=True)
                        t.start()
                        print(f"[watchdog] restarted '{name}'", flush=True)
        except Exception as e:
            print(f"[watchdog] {e}", flush=True)
        time.sleep(15)
