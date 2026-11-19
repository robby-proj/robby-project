"""
input_controller.py - Robodog Pro v8.4.6 production USB gamepad service.

SHANWAN 2563:0526 / PS2-STYLE CONTROL MAP

    CONTROL                    ROBODOG RESULT
    -------------------------  --------------------------------------------
    Left stick up/down         Forward / backward
    Left stick left/right      Strafe left / right
    Right stick left/right     Turn left / right
    D-pad                      Forward / backward / strafe left / right
    L3 or R3 stick click       HARD STOP; cancel autonomy and walkaround
    Cross / X                  Sit Beg              (preset 18)
    Circle                     Dance                (preset 16)
    Square                     Creep Forward        (preset 3)
    Triangle                   Look for Food        (preset 10)
    L1 or L2                   Start Modulino IMU autonomous mode
    R1 or R2                   Start dog-like walkaround
    Select                     Battery query
    Start                      Start dog-like walkaround
    Mode / Home                HARD STOP

V8 SAFETY ARCHITECTURE

    Input reader -> state/hysteresis -> latest-command slot -> command worker

    * The input reader never calls RouterBridge and therefore cannot block.
    * Only one reader may own /dev/input/js0.
    * A controller must be neutral before movement is armed.
    * Hard STOP, disconnect, collision, or Bridge loss disarms movement.
    * Re-arming requires all movement controls neutral for a stable interval.
    * Separate activation/release thresholds prevent dead-zone chatter.
    * Buttons are debounced; mode buttons have a longer lockout.
    * Manual input takes ownership from autonomous/walkaround exactly once.
    * Bridge loss discards stale motion. Bridge recovery sends STOP and rearms.
    * Modulino impact/tilt latches a safety STOP and blocks movement until clear.
    * Initialization events update state but never command the dog.
    * Trigger axes 4 and 5 are ignored; SHANWAN reports -32767 at rest.
    * Runtime state is memory-only. No controller data is written to disk.

APP LAB CONTAINER REQUIREMENTS

    devices:
      - /dev/input/js0:/dev/input/js0:rwm
      - /dev/input/event2:/dev/input/event2:rwm
    device_cgroup_rules:
      - "c 13:* rmw"
"""

from __future__ import annotations

import errno
import glob
import os
import queue
import select
import socket
import struct
import threading
import time
from typing import Dict, Iterator, Optional, Tuple

import state as S
import autonomy as A
try:
    import audio_feedback as F
except Exception:
    F = None


# Linux joystick ABI: u32 time, s16 value, u8 type, u8 number.
_JS = struct.Struct("IhBB")
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80

# Linux evdev input_event on 64-bit ARM. value is signed s32.
_EV = struct.Struct("@llHHi")
EV_KEY = 0x01
EV_ABS = 0x03

# Common evdev buttons.
BTN_SOUTH = 304
BTN_EAST = 305
BTN_NORTH = 307
BTN_WEST = 308
BTN_TL = 310
BTN_TR = 311
BTN_TL2 = 312
BTN_TR2 = 313
BTN_SELECT = 314
BTN_START = 315
BTN_MODE = 316
BTN_THUMBL = 317
BTN_THUMBR = 318

# Common evdev axes.
ABS_X = 0
ABS_Y = 1
ABS_RX = 3
ABS_RY = 4
ABS_HAT0X = 16
ABS_HAT0Y = 17

# Timing and control tuning.
AXIS_ACTIVATE = 0.32
AXIS_RELEASE = 0.18
NEUTRAL_ARM_SEC = 0.45
MOVE_RESEND_SEC = 0.35
COMMAND_TICK_SEC = 0.05
DEVICE_RESCAN_SEC = 1.0
DISCONNECT_STOP_SEC = 0.5
SELECT_TIMEOUT_SEC = 0.5
READ_CHUNK_SIZE = 4096
BUTTON_DEBOUNCE_SEC = 0.25
MODE_BUTTON_LOCKOUT_SEC = 1.0
SAFETY_CLEAR_SEC = 1.0
EVENT_LOG_EVERY = 200
ACTION_QUEUE_SIZE = 16

# Default collision limits. state.py values take precedence when present.
DEFAULT_IMPACT_VIBRATION = 1.20
DEFAULT_TILT_G = 0.70

# Reader lifecycle and command-worker lifecycle.
_rescan = threading.Event()
_reader_lock = threading.Lock()
_worker_start_lock = threading.Lock()
_worker_started = False
_reader_generation = 0

# Input and desired-command state.
_axis_lock = threading.Lock()
_command_lock = threading.Lock()
_button_lock = threading.Lock()
_status_lock = threading.Lock()
_wake_command_worker = threading.Event()
_hard_stop_requested = threading.Event()
_action_queue: queue.Queue = queue.Queue(maxsize=ACTION_QUEUE_SIZE)

_axes = {
    "x": 0.0,
    "y": 0.0,
    "turn": 0.0,
    "right_y": 0.0,
    "hat_x": 0.0,
    "hat_y": 0.0,
}

_desired_command = "stop"
_desired_generation = 0
_hard_stop_reason = "controller hard stop"
_last_button_time: Dict[str, float] = {}
_neutral_since: Optional[float] = None
_bridge_was_ok = False
_safety_clear_since: Optional[float] = None

_safety_unsafe_count = 0
_safety_safe_count = 0


# ---------------------------------------------------------------------------
# Runtime status
# ---------------------------------------------------------------------------
def _cs() -> Dict:
    st = S.STATE.setdefault("controller", {})
    defaults = {
        "enabled": True,
        "connected": False,
        "receiver_present": False,
        "controller_active": False,
        "init_events": 0,
        "device": None,
        "api": None,
        "name": None,
        "permission_ok": None,
        "last_event": None,
        "last_event_time": 0,
        "last_button": None,
        "last_button_name": None,
        "last_action": None,
        "last_command": "stop",
        "desired_command": "stop",
        "events": 0,
        "reconnects": 0,
        "disconnects": 0,
        "error": None,
        "cgroup_hint": None,
        "armed": False,
        "hard_stop_latched": True,
        "owner": "stopped",
        "bridge_available": False,
        "safety_blocked": False,
        "safety_reason": None,
        "reader_active": False,
        "reader_generation": 0,
        "command_worker_alive": False,
        "axis_state": dict(_axes),
        "mapping": "shanwan-2563-0526-ps2-v8.4.6",
    }
    for key, value in defaults.items():
        st.setdefault(key, value)
    return st


def status() -> Dict:
    st = dict(_cs())
    with _axis_lock:
        st["axis_state"] = {k: round(v, 3) for k, v in _axes.items()}
    return st


def request_rescan() -> None:
    _rescan.set()


def set_enabled(enabled: bool) -> None:
    st = _cs()
    st["enabled"] = bool(enabled)
    S.STATE["gamepad_enabled"] = bool(enabled)
    if not enabled:
        _request_hard_stop("controller disabled")
    request_rescan()


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------
def _sysfs_name(node: str) -> Optional[str]:
    base = os.path.basename(os.path.realpath(node))
    try:
        with open(
            f"/sys/class/input/{base}/device/name",
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:
            return file.read().strip() or None
    except Exception:
        return None


def _candidate_devices() -> Iterator[str]:
    """Prefer stable js nodes; use ABS-capable evdev only as fallback."""
    seen = set()
    candidates = []

    for path in sorted(glob.glob("/dev/input/by-id/*-joystick")):
        if "-event-joystick" not in path:
            candidates.append(path)
    candidates.extend(sorted(glob.glob("/dev/input/js*")))
    candidates.extend(sorted(glob.glob("/dev/input/by-id/*-event-joystick")))
    candidates.extend(sorted(glob.glob("/dev/input/event*")))

    for path in candidates:
        real = os.path.realpath(path)
        if real in seen:
            continue
        base = os.path.basename(real)
        if base.startswith("event") and "joystick" not in path:
            caps = f"/sys/class/input/{base}/device/capabilities/abs"
            try:
                with open(caps, "r", encoding="ascii") as file:
                    if not int(file.read().strip() or "0", 16):
                        continue
            except Exception:
                continue
        seen.add(real)
        yield path


def _open_candidate(path: str) -> Tuple[Optional[int], Optional[str]]:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except PermissionError as exc:
        return None, "cgroup" if exc.errno == errno.EPERM else "permission"
    except OSError:
        return None, None
    real = os.path.realpath(path)
    return fd, "js" if os.path.basename(real).startswith("js") else "evdev"


# ---------------------------------------------------------------------------
# Axis state, hysteresis, neutral arming
# ---------------------------------------------------------------------------
def _normalize_js(value: int) -> float:
    return max(-1.0, min(1.0, float(value) / 32767.0))


def _normalize_evdev_fallback(value: int) -> float:
    value = int(value)
    if 0 <= value <= 255:
        return max(-1.0, min(1.0, (value - 127.5) / 127.5))
    return max(-1.0, min(1.0, value / 32767.0))


def _reset_axes() -> None:
    global _neutral_since
    with _axis_lock:
        for key in _axes:
            _axes[key] = 0.0
    _neutral_since = time.monotonic()
    _publish_axes()


def _publish_axes() -> None:
    st = _cs()
    with _axis_lock:
        st["axis_state"] = {k: round(v, 3) for k, v in _axes.items()}


def _all_movement_controls_neutral() -> bool:
    with _axis_lock:
        return all(
            abs(_axes[key]) <= AXIS_RELEASE
            for key in ("x", "y", "turn", "right_y", "hat_x", "hat_y")
        )


def _update_arming() -> None:
    global _neutral_since
    st = _cs()
    now = time.monotonic()

    blocked = bool(st.get("safety_blocked")) or not bool(st.get("bridge_available"))
    if blocked or not st.get("controller_active") or not st.get("enabled", True):
        st["armed"] = False
        _neutral_since = None
        return

    if not _all_movement_controls_neutral():
        _neutral_since = None
        return

    if _neutral_since is None:
        _neutral_since = now
        return

    if now - _neutral_since >= NEUTRAL_ARM_SEC:
        if not st.get("armed"):
            print("[gamepad-safety] neutral confirmed; movement armed", flush=True)
        st["armed"] = True
        st["hard_stop_latched"] = False
        if st.get("owner") in ("stopped", "safety"):
            st["owner"] = "manual"


def _axis_command_hysteresis() -> str:
    """Choose one command using activation/release hysteresis."""
    st = _cs()
    previous = str(st.get("desired_command", "stop"))

    with _axis_lock:
        x = _axes["hat_x"] if abs(_axes["hat_x"]) > AXIS_RELEASE else _axes["x"]
        y = _axes["hat_y"] if abs(_axes["hat_y"]) > AXIS_RELEASE else _axes["y"]
        turn = _axes["turn"]

    # Preserve an active command until its source falls below release threshold.
    held = {
        "turn_left": turn < -AXIS_RELEASE,
        "turn_right": turn > AXIS_RELEASE,
        "forward": y < -AXIS_RELEASE,
        "backward": y > AXIS_RELEASE,
        "left": x < -AXIS_RELEASE,
        "right": x > AXIS_RELEASE,
    }
    if previous in held and held[previous]:
        return previous

    if turn < -AXIS_ACTIVATE:
        return "turn_left"
    if turn > AXIS_ACTIVATE:
        return "turn_right"

    if abs(y) >= abs(x):
        if y < -AXIS_ACTIVATE:
            return "forward"
        if y > AXIS_ACTIVATE:
            return "backward"
    else:
        if x < -AXIS_ACTIVATE:
            return "left"
        if x > AXIS_ACTIVATE:
            return "right"
    return "stop"


# ---------------------------------------------------------------------------
# Latest-command slot and action queue
# ---------------------------------------------------------------------------
def _set_desired_command(command: str) -> None:
    global _desired_command, _desired_generation
    st = _cs()

    if command != "stop" and not st.get("armed"):
        command = "stop"

    with _command_lock:
        if command == _desired_command:
            return
        _desired_command = command
        _desired_generation += 1
        st["desired_command"] = command
    _wake_command_worker.set()


def _get_desired_command() -> Tuple[str, int]:
    with _command_lock:
        return _desired_command, _desired_generation


def _clear_pending_actions() -> None:
    while True:
        try:
            _action_queue.get_nowait()
        except queue.Empty:
            return
        else:
            _action_queue.task_done()


def _enqueue_action(kind: str, value: Optional[str] = None) -> None:
    try:
        _action_queue.put_nowait((kind, value, time.monotonic()))
        _wake_command_worker.set()
    except queue.Full:
        print(f"[gamepad-action] queue full; dropped {kind}:{value}", flush=True)


def _request_hard_stop(reason: str) -> None:
    global _hard_stop_reason
    st = _cs()
    _hard_stop_reason = str(reason)
    st["armed"] = False
    st["hard_stop_latched"] = True
    st["owner"] = "stopped"
    _set_desired_command("stop")
    _clear_pending_actions()
    _hard_stop_requested.set()
    _wake_command_worker.set()


# ---------------------------------------------------------------------------
# Button debounce and PS2 mapping
# ---------------------------------------------------------------------------
def _button_allowed(logical: str) -> bool:
    now = time.monotonic()
    interval = (
        MODE_BUTTON_LOCKOUT_SEC
        if logical in {"l1", "l2", "r1", "r2", "start", "mode", "l3", "r3"}
        else BUTTON_DEBOUNCE_SEC
    )
    with _button_lock:
        last = _last_button_time.get(logical, 0.0)
        if now - last < interval:
            return False
        _last_button_time[logical] = now
    return True


def _handle_button(logical: str, code: int) -> None:
    if not logical or not _button_allowed(logical):
        return

    st = _cs()
    st["last_button"] = int(code)
    st["last_button_name"] = logical

    if logical in ("l3", "r3", "mode"):
        print(f"[gamepad-safety] {logical.upper()} -> HARD STOP", flush=True)
        _request_hard_stop(f"{logical} hard stop")
        return

    mapping = {
        "cross": ("action", "sit_beg"),
        "circle": ("action", "dance"),
        "square": ("action", "creep_fwd"),
        "triangle": ("action", "look_food"),
        "l1": ("autonomous", None),
        "l2": ("autonomous", None),
        "r1": ("walkaround", None),
        "r2": ("walkaround", None),
        "select": ("battery", None),
        "start": ("walkaround", None),
    }
    job = mapping.get(logical)
    if job:
        print(f"[gamepad-button] {logical} -> {job[0]}{':' + job[1] if job[1] else ''}", flush=True)
        _enqueue_action(*job)


# ---------------------------------------------------------------------------
# Input event handlers. These never call RouterBridge.
# ---------------------------------------------------------------------------
def _handle_js(number: int, value: int, event_type: int) -> None:
    is_initial = bool(event_type & JS_EVENT_INIT)
    kind = event_type & ~JS_EVENT_INIT

    if kind == JS_EVENT_BUTTON:
        if is_initial or value != 1:
            return
        buttons = {
            0: "cross",
            1: "circle",
            2: "square",
            3: "triangle",
            4: "l1",
            5: "r1",
            6: "l2",
            7: "r2",
            8: "select",
            9: "start",
            10: "mode",
            11: "l3",
            12: "r3",
        }
        _handle_button(buttons.get(number, ""), number)
        return

    if kind != JS_EVENT_AXIS:
        return

    normalized = _normalize_js(value)
    with _axis_lock:
        if number == 0:
            _axes["x"] = normalized
        elif number == 1:
            _axes["y"] = normalized
        elif number == 2:
            _axes["turn"] = normalized
        elif number == 3:
            _axes["right_y"] = normalized
        elif number == 6:
            _axes["hat_x"] = normalized
        elif number == 7:
            _axes["hat_y"] = normalized
        else:
            return  # axes 4/5 are analog triggers
    _publish_axes()

    if is_initial:
        return

    _update_arming()
    _set_desired_command(_axis_command_hysteresis())


def _handle_evdev(code: int, value: int, event_type: int) -> None:
    if event_type == EV_KEY:
        if value != 1:
            return
        buttons = {
            BTN_SOUTH: "cross",
            BTN_EAST: "circle",
            BTN_NORTH: "square",
            BTN_WEST: "triangle",
            BTN_TL: "l1",
            BTN_TR: "r1",
            BTN_TL2: "l2",
            BTN_TR2: "r2",
            BTN_SELECT: "select",
            BTN_START: "start",
            BTN_MODE: "mode",
            BTN_THUMBL: "l3",
            BTN_THUMBR: "r3",
        }
        _handle_button(buttons.get(code, ""), code)
        return

    if event_type != EV_ABS:
        return
    if code in (ABS_HAT0X, ABS_HAT0Y):
        normalized = -1.0 if value < 0 else 1.0 if value > 0 else 0.0
    else:
        normalized = _normalize_evdev_fallback(value)
    with _axis_lock:
        if code == ABS_X:
            _axes["x"] = normalized
        elif code == ABS_Y:
            _axes["y"] = normalized
        elif code == ABS_RX:
            _axes["turn"] = normalized
        elif code == ABS_RY:
            _axes["right_y"] = normalized
        elif code == ABS_HAT0X:
            _axes["hat_x"] = normalized
        elif code == ABS_HAT0Y:
            _axes["hat_y"] = normalized
        else:
            return
    _publish_axes()
    _update_arming()
    _set_desired_command(_axis_command_hysteresis())


def _record_event(api: str, event_type: int, code: int, value: int) -> None:
    st = _cs()
    st["last_event"] = {
        "api": api,
        "type": int(event_type),
        "code": int(code),
        "value": int(value),
    }
    st["last_event_time"] = int(time.time())
    st["events"] = int(st.get("events", 0)) + 1
    count = st["events"]
    # Production logging: live events only, plus sparse heartbeat samples.
    if not (event_type & JS_EVENT_INIT) and (count <= 50 or count % EVENT_LOG_EVERY == 0):
        print(f"[gamepad-input] {st['last_event']}", flush=True)


# ---------------------------------------------------------------------------
# Command ownership, Bridge health, and collision safety
# ---------------------------------------------------------------------------
def _sequence_running() -> bool:
    seq = S.STATE.get("sequence", {})
    return isinstance(seq, dict) and bool(seq.get("running"))


def _autonomous_mode_active() -> bool:
    return S.STATE.get("mode") in {"autonomous_imu", "autonomous_vision"}


def _manual_takeover() -> None:
    st = _cs()
    if _sequence_running():
        A.stop_walkaround_sequence()
    if _autonomous_mode_active():
        A.set_mode("manual")
    st["owner"] = "manual"


def _perform_hard_stop(reason: str) -> None:
    st = _cs()
    try:
        if _sequence_running():
            A.stop_walkaround_sequence()
        if _autonomous_mode_active():
            A.set_mode("manual")
        else:
            A.send_command("stop", force=True, announce=False)
    except Exception as exc:
        print(f"[gamepad-safety] STOP error: {exc}", flush=True)
    _reset_axes()
    st.update(
        armed=False,
        hard_stop_latched=True,
        owner="stopped",
        last_command="stop",
        desired_command="stop",
        safety_reason=reason if st.get("safety_blocked") else st.get("safety_reason"),
    )
    print(f"[gamepad-safety] HARD STOP: {reason}", flush=True)


def _safety_sample() -> Tuple[bool, Optional[str]]:
    motion = getattr(S, "MOTION", {})
    vibration = float(motion.get("vibration", 0.0) or 0.0)
    x = float(motion.get("x", 0.0) or 0.0)
    y = float(motion.get("y", 0.0) or 0.0)
    impact_limit = float(getattr(S, "AUTONOMY_IMPACT_VIBRATION", DEFAULT_IMPACT_VIBRATION))
    tilt_limit = float(getattr(S, "AUTONOMY_TILT_G", DEFAULT_TILT_G))
    if vibration >= impact_limit:
        return True, f"impact vibration={vibration:.3f}"
    if abs(x) >= tilt_limit or abs(y) >= tilt_limit:
        return True, f"tilt x={x:.3f} y={y:.3f}"
    return False, None


def _motion_producer_active() -> bool:
    """True only when the physical controller currently owns manual motion.

    Autonomous IMU, autonomous vision and scripted sequences have their own
    confirmed safety policies. Letting the disconnected controller worker
    supervise those owners caused a mounted sensor's normal gravity vector to
    force every autonomous mode back to manual.
    """
    st = _cs()
    return bool(
        st.get("enabled")
        and st.get("connected")
        and st.get("controller_active")
        and st.get("armed")
        and st.get("owner") == "manual"
        and st.get("last_command") not in (None, "stop")
    )


def _update_collision_safety() -> None:
    global _safety_clear_since, _safety_unsafe_count, _safety_safe_count
    st = _cs()
    unsafe, reason = _safety_sample()
    now = time.monotonic()
    active_required = bool(getattr(S, "AUTONOMY_SAFETY_ONLY_WHILE_ACTIVE", True))

    # Never let the gamepad safety worker seize ownership from another producer.
    # Autonomy and sequence safety are handled by autonomy.py.
    if _autonomous_mode_active() or _sequence_running():
        unsafe = False
        reason = None

    if active_required and not _motion_producer_active():
        unsafe = False
        reason = None
    if unsafe:
        _safety_unsafe_count += 1
        _safety_safe_count = 0
    else:
        _safety_safe_count += 1
        _safety_unsafe_count = 0
    confirm = int(getattr(S, "AUTONOMY_SAFETY_CONFIRM_SAMPLES", 5))
    unsafe = unsafe and _safety_unsafe_count >= max(1, confirm)

    if unsafe and _motion_producer_active():
        _safety_clear_since = None
        if not st.get("safety_blocked"):
            st["safety_blocked"] = True
            st["safety_reason"] = reason
            _request_hard_stop(reason or "collision safety")
        return

    if not st.get("safety_blocked"):
        return

    if unsafe:
        _safety_clear_since = None
        return

    if _safety_clear_since is None:
        _safety_clear_since = now
        return

    if now - _safety_clear_since >= SAFETY_CLEAR_SEC:
        st["safety_blocked"] = False
        st["safety_reason"] = None
        st["armed"] = False
        st["hard_stop_latched"] = True
        _safety_clear_since = None
        print("[gamepad-safety] IMU stable; return controls to neutral to re-arm", flush=True)


def _bridge_ok() -> bool:
    bridge = getattr(S, "BRIDGE_STATE", {})
    return bool(bridge.get("ok"))


def _process_action(kind: str, value: Optional[str]) -> None:
    st = _cs()
    if st.get("safety_blocked") or not st.get("bridge_available"):
        print(f"[gamepad-action] blocked {kind}: safety/Bridge unavailable", flush=True)
        return

    if kind == "action" and value:
        _manual_takeover()
        st["owner"] = "preset_action"
        st["last_action"] = value
        A.send_action_named(value)
    elif kind == "autonomous":
        if _sequence_running():
            A.stop_walkaround_sequence()
        if S.STATE.get("mode") != "autonomous_imu":
            A.set_mode("autonomous_imu")
        st["owner"] = "imu"
        st["last_action"] = "autonomous_imu"
    elif kind == "walkaround":
        if _autonomous_mode_active():
            A.set_mode("manual")
        if not _sequence_running():
            A.start_walkaround_sequence()
        st["owner"] = "walkaround"
        st["last_action"] = "walkaround"
    elif kind == "battery":
        A.send_command("battery")
        st["last_action"] = "battery"


def _controller_command_worker() -> None:
    global _bridge_was_ok
    st = _cs()
    st["command_worker_alive"] = True
    last_sent = "stop"
    last_sent_at = 0.0
    last_generation = -1

    while True:
        st["command_worker_alive"] = True
        _update_collision_safety()

        bridge_now = _bridge_ok()
        st["bridge_available"] = bridge_now
        if bridge_now != _bridge_was_ok:
            if bridge_now:
                print("[gamepad-bridge] Bridge recovered; STOP and neutral re-arm required", flush=True)
                st["armed"] = False
                st["hard_stop_latched"] = True
                _set_desired_command("stop")
                try:
                    A.send_command("stop", force=True, announce=False)
                except Exception as exc:
                    print(f"[gamepad-bridge] recovery STOP failed: {exc}", flush=True)
            else:
                print("[gamepad-bridge] Bridge unavailable; stale motion discarded", flush=True)
                st["armed"] = False
                st["hard_stop_latched"] = True
                _set_desired_command("stop")
            _bridge_was_ok = bridge_now

        if _hard_stop_requested.is_set():
            _hard_stop_requested.clear()
            if bridge_now:
                _perform_hard_stop(_hard_stop_reason)
            else:
                _reset_axes()
            last_sent = "stop"
            last_sent_at = time.monotonic()
            st["last_command"] = "stop"

        _update_arming()

        # Mode/actions are serialized through the same worker as locomotion.
        try:
            kind, value, _queued_at = _action_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            try:
                _process_action(kind, value)
            except Exception as exc:
                print(f"[gamepad-action] {kind}:{value} failed: {exc}", flush=True)
            finally:
                _action_queue.task_done()

        command, generation = _get_desired_command()
        now = time.monotonic()
        should_send = generation != last_generation
        if command != "stop" and now - last_sent_at >= MOVE_RESEND_SEC:
            should_send = True

        if should_send:
            last_generation = generation
            if not bridge_now:
                pass  # never replay movement requested during Bridge loss
            elif command != "stop" and (not st.get("armed") or st.get("safety_blocked")):
                pass
            else:
                try:
                    if command != "stop":
                        _manual_takeover()
                    if command != last_sent or command != "stop":
                        A.send_command(command)
                        print(f"[gamepad-command] {command}", flush=True)
                    last_sent = command
                    last_sent_at = now
                    st["last_command"] = command
                    st["owner"] = "manual" if command != "stop" else st.get("owner", "manual")
                except Exception as exc:
                    print(f"[gamepad-command] {command} failed: {exc}", flush=True)

        _wake_command_worker.wait(COMMAND_TICK_SEC)
        _wake_command_worker.clear()


def _ensure_command_worker() -> None:
    global _worker_started
    with _worker_start_lock:
        if _worker_started:
            return
        thread = threading.Thread(
            target=_controller_command_worker,
            name="gamepad-command-worker",
            daemon=True,
        )
        thread.start()
        _worker_started = True


# ---------------------------------------------------------------------------
# Single-owner reader loop
# ---------------------------------------------------------------------------
def controller_loop() -> None:
    """Single-owner TCP consumer for the isolated gamepad-reader Brick."""
    global _reader_generation, _neutral_since
    _ensure_command_worker()
    if not _reader_lock.acquire(blocking=False):
        print("[gamepad-device] duplicate reader start ignored", flush=True)
        return

    st = _cs()
    _reader_generation += 1
    st.update(reader_active=True, reader_generation=_reader_generation)
    try:
        while True:
            S.heartbeat("gamepad")
            if not st.get("enabled", True):
                st.update(connected=False, receiver_present=False, controller_active=False,
                          device=None, api=None, error=None)
                _rescan.wait(DEVICE_RESCAN_SEC); _rescan.clear(); continue

            sock = None
            try:
                sock = socket.create_connection(("gamepad-reader", 8765), timeout=3.0)
                sock.settimeout(None)
                recv_buffer = bytearray()
                _reset_axes(); _neutral_since = time.monotonic()
                st.update(connected=False, receiver_present=False, controller_active=False,
                          device="gamepad-reader", api="sidecar-js",
                          name="SHANWAN Android Gamepad", permission_ok=True, error=None,
                          cgroup_hint=None, armed=False, hard_stop_latched=True, owner="stopped")
                print("[gamepad-device] sidecar connected; waiting for joystick", flush=True)

                while not _rescan.is_set() and st.get("enabled", True):
                    S.heartbeat("gamepad"); _update_arming()
                    while b"\n" not in recv_buffer:
                        chunk = sock.recv(4096)
                        if not chunk: raise OSError("gamepad sidecar disconnected")
                        recv_buffer.extend(chunk)
                    raw_line, _, rest = recv_buffer.partition(b"\n")
                    recv_buffer = bytearray(rest)
                    parts = raw_line.decode("utf-8", "replace").split("\t")
                    if not parts or parts[0] == "PING": continue
                    if parts[0] == "STATUS":
                        if len(parts) >= 2 and parts[1] == "connected":
                            device = parts[2] if len(parts) > 2 else "/dev/input/js0"
                            st.update(receiver_present=True, connected=False, controller_active=False,
                                      device=device, error="receiver detected; controller inactive")
                            S.STATE["gamepad"] = device
                        else:
                            st.update(receiver_present=False, connected=False, controller_active=False,
                                      error="sidecar waiting for receiver")
                        continue
                    if len(parts) == 4 and parts[0] == "JS":
                        event_type, number, value = map(int, parts[1:])
                        base_type = event_type & ~JS_EVENT_INIT
                        if base_type not in (JS_EVENT_BUTTON, JS_EVENT_AXIS):
                            continue
                        if event_type & JS_EVENT_INIT:
                            st["init_events"] = int(st.get("init_events", 0)) + 1
                            continue
                        if not st.get("controller_active"):
                            st.update(connected=True, controller_active=True, error=None,
                                      armed=False, hard_stop_latched=True)
                            st["reconnects"] = int(st.get("reconnects", 0)) + 1
                            _reset_axes(); _neutral_since = time.monotonic()
                            if F: F.emit("controller_connected")
                            print("[gamepad-device] controller activity confirmed; neutral required", flush=True)
                        _record_event("sidecar-js", event_type, number, value)
                        _handle_js(number, value, event_type)
            except Exception as exc:
                st.update(connected=False, receiver_present=False, controller_active=False,
                          device=None, api="sidecar-js", permission_ok=None,
                          error=f"gamepad sidecar unavailable: {exc}", armed=False,
                          hard_stop_latched=True)
                S.STATE["gamepad"] = st["error"]
                print(f"[gamepad-device] {st['error']}", flush=True)
            finally:
                was_active = bool(st.get("controller_active"))
                was_moving = str(st.get("last_command", "stop")) != "stop"
                intentional = (not st.get("enabled", True)) or _rescan.is_set()
                if sock:
                    try: sock.close()
                    except Exception: pass
                st.update(connected=False, receiver_present=False, controller_active=False,
                          device=None, armed=False, hard_stop_latched=True,
                          disconnects=int(st.get("disconnects", 0)) + 1)
                _rescan.clear()
                if not intentional and (was_active or was_moving):
                    _request_hard_stop("active controller disconnected")
                time.sleep(max(DISCONNECT_STOP_SEC, 2.0))
    finally:
        st["reader_active"] = False
        _reader_lock.release()

