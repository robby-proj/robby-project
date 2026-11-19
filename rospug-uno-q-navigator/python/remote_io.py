"""App Lab sidecar clients for the ROSPug gamepad and LD19 LiDAR."""

from __future__ import annotations

import json
import os
import socket
import threading
import time


GAMEPAD_HOST = os.getenv("ROSPUG_GAMEPAD_BRICK_HOST", "gamepad-reader")
GAMEPAD_PORT = int(os.getenv("ROSPUG_GAMEPAD_BRICK_PORT", "8765"))
LIDAR_HOST = os.getenv("ROSPUG_LIDAR_BRICK_HOST", "lidar-reader")
LIDAR_PORT = int(os.getenv("ROSPUG_LIDAR_BRICK_PORT", "8766"))

AXES = ("lx", "ly", "rx", "ry", "trigger_r", "trigger_l", "hat_x", "hat_y")
BUTTONS = {
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


def _empty_axes() -> dict:
    return {name: 0.0 for name in AXES}


def _empty_buttons() -> dict:
    return {name: False for name in BUTTONS.values()}


def _empty_lidar(error: str) -> dict:
    return {
        "online": False,
        "fresh": False,
        "path": "",
        "baud": 230400,
        "error": error,
        "valid_packets": 0,
        "crc_errors": 0,
        "last_packet_age_ms": -1,
        "speed_dps": 0,
        "front_offset_deg": 0.0,
        "sectors": {},
    }


class GamepadReader:
    """Consume Linux joystick events from the isolated gamepad_input Brick."""

    JS_EVENT_BUTTON = 0x01
    JS_EVENT_AXIS = 0x02
    JS_EVENT_INIT = 0x80

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rospug-gamepad", daemon=True)
        self._started = False
        self._axes = _empty_axes()
        self._buttons = _empty_buttons()
        self._receiver_present = False
        self._controller_active = False
        self._armed = False
        self._neutral_since = None
        self._online = False
        self._path = ""
        self._error = "Waiting for gamepad_input Brick"
        self._events = 0
        self._init_events = 0
        self._last_event = 0.0

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> dict:
        with self._lock:
            age_ms = int((time.monotonic() - self._last_event) * 1000) if self._last_event else -1
            return {
                "online": self._online,
                "receiver_present": self._receiver_present,
                "controller_active": self._controller_active,
                "armed": self._armed,
                "path": self._path,
                "api": "sidecar-js",
                "error": self._error,
                "events": self._events,
                "init_events": self._init_events,
                "last_event_age_ms": age_ms,
                "axes": dict(self._axes),
                "buttons": dict(self._buttons),
            }

    def _disconnect(self, error: str) -> None:
        with self._lock:
            self._receiver_present = False
            self._controller_active = False
            self._armed = False
            self._neutral_since = None
            self._online = False
            self._path = ""
            self._error = error
            self._axes = _empty_axes()
            self._buttons = _empty_buttons()

    def _update_arming_locked(self, now: float) -> None:
        if self._armed or not self._controller_active:
            return
        movement_axes = ("lx", "ly", "rx", "ry", "hat_x", "hat_y")
        neutral = all(abs(self._axes[name]) <= 0.18 for name in movement_axes)
        if not neutral:
            self._neutral_since = None
            self._error = "Return all movement controls to neutral"
            return
        if self._neutral_since is None:
            self._neutral_since = now
            self._error = "Neutral detected; arming gamepad"
            return
        if now - self._neutral_since >= 0.45:
            self._armed = True
            self._online = True
            self._error = ""

    def _handle_line(self, line: str) -> None:
        parts = line.split("\t")
        if not parts:
            return
        if parts[0] == "PING":
            with self._lock:
                self._update_arming_locked(time.monotonic())
            return
        if parts[0] == "STATUS":
            with self._lock:
                if len(parts) >= 2 and parts[1] == "connected":
                    self._receiver_present = True
                    self._path = parts[2] if len(parts) > 2 else "/dev/input/js0"
                    if not self._online:
                        self._error = "Receiver detected; move a control once to arm input"
                else:
                    self._receiver_present = False
                    self._controller_active = False
                    self._armed = False
                    self._online = False
                    self._path = ""
                    self._error = "Gamepad receiver not found"
            return
        if len(parts) != 4 or parts[0] != "JS":
            return

        event_type, number, value = map(int, parts[1:])
        initial = bool(event_type & self.JS_EVENT_INIT)
        kind = event_type & ~self.JS_EVENT_INIT
        now = time.monotonic()
        with self._lock:
            if kind == self.JS_EVENT_AXIS and 0 <= number < len(AXES):
                # Trigger axes 4/5 are diagnostic only: this receiver reports
                # -32767 at rest, so they must not become movement controls.
                self._axes[AXES[number]] = max(-1.0, min(1.0, value / 32767.0))
            elif kind == self.JS_EVENT_BUTTON and number in BUTTONS:
                self._buttons[BUTTONS[number]] = bool(value)
            else:
                return
            if initial:
                self._init_events += 1
                return
            self._controller_active = True
            self._receiver_present = True
            self._events += 1
            self._last_event = now
            self._update_arming_locked(now)

    def _run(self) -> None:
        while not self._stop.is_set():
            sock = None
            stream = None
            try:
                sock = socket.create_connection((GAMEPAD_HOST, GAMEPAD_PORT), timeout=3.0)
                sock.settimeout(None)
                stream = sock.makefile("r", encoding="utf-8", errors="replace", newline="\n")
                for raw in stream:
                    if self._stop.is_set():
                        return
                    self._handle_line(raw.rstrip("\r\n"))
                raise OSError("gamepad_input Brick disconnected")
            except (OSError, ValueError) as exc:
                self._disconnect(f"Gamepad Brick unavailable: {exc}")
            finally:
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            self._stop.wait(1.0)


class LD19Reader:
    """Consume validated sector snapshots from the isolated lidar_input Brick."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rospug-ld19", daemon=True)
        self._started = False
        self._snapshot = _empty_lidar("Waiting for lidar_input Brick")
        self._last_snapshot = 0.0

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> dict:
        with self._lock:
            result = json.loads(json.dumps(self._snapshot))
            age = time.monotonic() - self._last_snapshot if self._last_snapshot else 999.0
        if age > 0.50:
            result["online"] = False
            result["fresh"] = False
            result["error"] = "LiDAR Brick snapshot is stale"
        return result

    def _disconnect(self, error: str) -> None:
        with self._lock:
            self._snapshot = _empty_lidar(error)
            self._last_snapshot = 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            sock = None
            stream = None
            try:
                sock = socket.create_connection((LIDAR_HOST, LIDAR_PORT), timeout=3.0)
                sock.settimeout(None)
                stream = sock.makefile("r", encoding="utf-8", errors="replace", newline="\n")
                for raw in stream:
                    if self._stop.is_set():
                        return
                    payload = json.loads(raw)
                    if not isinstance(payload, dict) or payload.get("protocol") != 1:
                        raise ValueError("invalid lidar_input Brick message")
                    lidar = payload.get("lidar")
                    if not isinstance(lidar, dict):
                        raise ValueError("missing LiDAR snapshot")
                    with self._lock:
                        self._snapshot = lidar
                        self._last_snapshot = time.monotonic()
                raise OSError("lidar_input Brick disconnected")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._disconnect(f"LiDAR Brick unavailable: {exc}")
            finally:
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            self._stop.wait(1.0)
