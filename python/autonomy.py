"""Motion arbitration and obstacle avoidance for the ROSPug UNO Q app."""

from __future__ import annotations

import threading
import time
from typing import Callable


MODES = ("manual", "guard", "auto")

# Auto Avoid deliberately uses a little more clearance than LiDAR Guard, but
# it must not require an obstacle return in every sector.  On a real LD19 an
# empty sector normally means that nothing reflective is in range.
AUTO_MIN_SCAN_POINTS = 18
AUTO_OPEN_DISTANCE = 12.0
AUTO_TRAPPED_DISTANCE = 0.22
AUTO_TURN_DISTANCE = 0.36
AUTO_CAUTION_DISTANCE = 0.70
AUTO_TURN_RELEASE_DISTANCE = 0.50
AUTO_SIDE_ROUTE_DISTANCE = 0.38
AUTO_REAR_ESCAPE_DISTANCE = 0.46
AUTO_REVERSE_STOP_DISTANCE = 0.30
AUTO_OVERRIDE_AXIS = 0.32

# Commands are refreshed about every 220 ms.  ROSPug's reverse command is a
# useful reference because it is a single, coordinated gait vector.  Apply the
# same fractional envelope to every axis so strafe, rotation, diagonals, and
# Auto Avoid do not ask different parts of the gait generator to change at
# different rates.  Safety stops still bypass this envelope and remain
# immediate.
COMMAND_LIMITS = (0.10, 0.08, 0.35)
COORDINATED_ACCEL_FRACTION = 0.25
COORDINATED_BRAKE_FRACTION = 0.40
SHARP_DIRECTION_COSINE = 0.35


def _deadzone(value: float, threshold: float = 0.18) -> float:
    if abs(value) <= threshold:
        return 0.0
    magnitude = (abs(value) - threshold) / (1.0 - threshold)
    return magnitude if value > 0 else -magnitude


def _nonzero(vector: tuple[float, float, float]) -> bool:
    return any(abs(value) > 0.001 for value in vector)


def _vector_size(vector: tuple[float, float, float]) -> float:
    return sum(value * value for value in vector) ** 0.5


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _slew_vector(
    current: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Blend one coordinated motion vector toward another.

    Components are normalized by their verified ROSPug limits before the
    common step is applied.  This makes a 25% strafe or rotation transition
    take the same time as a 25% reverse transition.  For a sharp change (for
    example forward directly to strafe), brake the old vector first instead
    of twisting the legs through both commands at once.
    """
    current_n = tuple(value / limit for value, limit in zip(current, COMMAND_LIMITS))
    target_n = tuple(value / limit for value, limit in zip(target, COMMAND_LIMITS))
    current_size = _vector_size(current_n)
    target_size = _vector_size(target_n)

    if current_size > 0.01 and target_size > 0.01:
        cosine = _dot(current_n, target_n) / (current_size * target_size)
        if cosine < SHARP_DIRECTION_COSINE:
            target_n = (0.0, 0.0, 0.0)
            target_size = 0.0

    delta = tuple(wanted - value for value, wanted in zip(current_n, target_n))
    largest_delta = max(abs(value) for value in delta)
    if largest_delta < 0.0001:
        return target

    slowing = target_size < current_size or _dot(current_n, target_n) < 0.0
    fraction = COORDINATED_BRAKE_FRACTION if slowing else COORDINATED_ACCEL_FRACTION
    scale = min(1.0, fraction / largest_delta)
    next_n = tuple(value + change * scale for value, change in zip(current_n, delta))
    result = tuple(value * limit for value, limit in zip(next_n, COMMAND_LIMITS))
    return tuple(0.0 if abs(value) < 0.001 else value for value in result)


class MotionSupervisor:
    """Combine browser, gamepad, and LiDAR inputs into one watchdog-fed command."""

    def __init__(
        self,
        gamepad,
        lidar,
        move_fn: Callable[[float, float, float], bool],
        stop_fn: Callable[[], bool],
        home_fn: Callable[[], bool],
        pose_fn: Callable[[float, float, float, float, float, float, float], bool],
    ) -> None:
        self.gamepad = gamepad
        self.lidar = lidar
        self._move_fn = move_fn
        self._stop_fn = stop_fn
        self._home_fn = home_fn
        self._pose_fn = pose_fn
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rospug-motion", daemon=True)

        self._mode = "guard"
        self._robot_connected = False
        self._robot_armed = False
        self._web_vector = (0.0, 0.0, 0.0)
        self._web_expires = 0.0
        self._desired = (0.0, 0.0, 0.0)
        self._commanded = (0.0, 0.0, 0.0)
        self._input_source = "none"
        self._blocked_reason = "Robot is not initialized"
        self._last_send = 0.0
        self._was_moving = False
        self._stop_requested = False
        self._stop_reason = ""
        self._previous_buttons = {}
        self._auto_turn = 0
        self._auto_turn_until = 0.0
        self._auto_reverse_until = 0.0
        self._auto_state = "idle"
        self._last_pose_send = 0.0
        self._pose = {
            "x": 0.0,
            "y": 0.0,
            "z": -0.13,
            "x_shift": 0.01,
            "roll": 0.0,
            "pitch": 0.0,
        }

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def update_robot_state(self, connected: bool, armed: bool) -> None:
        with self._lock:
            self._robot_connected = bool(connected)
            self._robot_armed = bool(armed)
            if not connected or not armed:
                self._web_vector = (0.0, 0.0, 0.0)
                self._web_expires = 0.0

    def set_web_intent(self, vector: tuple[float, float, float]) -> None:
        now = time.monotonic()
        with self._lock:
            if self._mode == "auto" and _nonzero(vector):
                self._mode = "guard"
                self._auto_turn = 0
                self._auto_reverse_until = 0.0
                self._auto_state = "idle"
            self._web_vector = vector
            self._web_expires = now + 0.60

    def clear_web_intent(self) -> None:
        with self._lock:
            self._web_vector = (0.0, 0.0, 0.0)
            self._web_expires = 0.0

    def set_mode(self, mode: str, confirmed: bool = False) -> tuple[bool, str]:
        if mode not in MODES:
            return False, "Unknown control mode"
        gamepad = self.gamepad.snapshot()
        lidar = self.lidar.snapshot()
        with self._lock:
            if mode == "auto":
                if not confirmed:
                    return False, "Auto Avoid requires explicit confirmation"
                if not (self._robot_connected and self._robot_armed):
                    return False, "Initialize the online robot before Auto Avoid"
                if not gamepad["online"]:
                    return False, "Auto Avoid requires the USB remote for override"
                if not lidar["fresh"]:
                    return False, "Auto Avoid requires fresh LD19 data"
            self._mode = mode
            self._web_vector = (0.0, 0.0, 0.0)
            self._web_expires = 0.0
            self._auto_turn = 0
            self._auto_reverse_until = 0.0
            self._auto_state = "starting" if mode == "auto" else "idle"
            self._stop_requested = True
            self._stop_reason = f"Changed to {mode} mode"
        labels = {"manual": "Manual", "guard": "LiDAR Guard", "auto": "Auto Avoid"}
        return True, f"{labels[mode]} mode selected"

    def emergency_stop(self, reason: str) -> None:
        with self._lock:
            self._mode = "guard"
            self._web_vector = (0.0, 0.0, 0.0)
            self._web_expires = 0.0
            self._auto_turn = 0
            self._auto_reverse_until = 0.0
            self._auto_state = "idle"
            self._stop_requested = True
            self._stop_reason = reason

    def snapshot(self) -> dict:
        gamepad = self.gamepad.snapshot()
        lidar = self.lidar.snapshot()
        with self._lock:
            return {
                "mode": self._mode,
                "input_source": self._input_source,
                "blocked_reason": self._blocked_reason,
                "desired": list(self._desired),
                "commanded": list(self._commanded),
                "auto_turn": self._auto_turn,
                "auto_state": self._auto_state,
                "pose": dict(self._pose),
                "gamepad": gamepad,
                "lidar": lidar,
            }

    @staticmethod
    def _remote_vector(gamepad: dict) -> tuple[float, float, float]:
        if not gamepad["online"]:
            return 0.0, 0.0, 0.0
        axes = gamepad["axes"]
        buttons = gamepad["buttons"]
        # Linux joystick values are negative for up/left. ROSPug uses positive
        # X for forward, positive Y for left, and positive yaw for left turn.
        x = -_deadzone(axes.get("ly", 0.0)) * 0.10
        y = -_deadzone(axes.get("lx", 0.0)) * 0.08
        yaw = -_deadzone(axes.get("rx", 0.0)) * 0.35

        if axes.get("hat_y", 0.0) < -0.5:
            x = 0.10
        elif axes.get("hat_y", 0.0) > 0.5:
            x = -0.10
        if axes.get("hat_x", 0.0) < -0.5:
            yaw = 0.35
        elif axes.get("hat_x", 0.0) > 0.5:
            yaw = -0.35
        if buttons.get("square"):
            y = 0.08
        elif buttons.get("circle"):
            y = -0.08
        return x, y, yaw

    @staticmethod
    def _auto_override_active(gamepad: dict) -> bool:
        """Require an intentional remote movement before cancelling Auto.

        The normal 0.18 deadzone is appropriate while driving, but some
        wireless controllers idle just outside it.  Auto therefore uses a
        larger takeover threshold while D-pad and strafe buttons remain
        immediate overrides.
        """
        if not gamepad.get("online"):
            return False
        axes = gamepad.get("axes", {})
        buttons = gamepad.get("buttons", {})
        stick_override = any(
            abs(float(axes.get(name, 0.0))) >= AUTO_OVERRIDE_AXIS
            for name in ("lx", "ly", "rx")
        )
        dpad_override = (
            abs(float(axes.get("hat_x", 0.0))) > 0.5
            or abs(float(axes.get("hat_y", 0.0))) > 0.5
        )
        return bool(stick_override or dpad_override or buttons.get("square") or buttons.get("circle"))

    @staticmethod
    def _scale_for_distance(value: float, distance: float | None, stop: float, caution: float) -> float:
        # Once the overall scan is known to be healthy, no return in this
        # direction means no obstacle was detected in range.
        if distance is None:
            return value
        if distance <= stop:
            return 0.0
        if distance >= caution:
            return value
        return value * (distance - stop) / (caution - stop)

    def _guard_vector(self, desired: tuple[float, float, float], lidar: dict) -> tuple[tuple[float, float, float], str]:
        if not lidar["fresh"]:
            return (0.0, 0.0, 0.0), "LiDAR data is missing or stale"
        scan_points = int(lidar.get("scan_points", 0) or 0)
        if scan_points < AUTO_MIN_SCAN_POINTS:
            return (
                (0.0, 0.0, 0.0),
                f"LiDAR Guard waiting: only {scan_points}/{AUTO_MIN_SCAN_POINTS} recent scan points",
            )
        sectors = lidar["sectors"]

        x, y, yaw = desired
        reasons = []
        if x > 0:
            limited = self._scale_for_distance(x, sectors.get("front"), 0.32, 0.75)
            if limited != x:
                reasons.append("forward limited by front clearance")
            x = limited
        elif x < 0:
            limited = self._scale_for_distance(x, sectors.get("rear"), 0.30, 0.65)
            if limited != x:
                reasons.append("reverse limited by rear clearance")
            x = limited

        if y > 0:
            limited = self._scale_for_distance(y, sectors.get("left"), 0.27, 0.58)
            if limited != y:
                reasons.append("left strafe limited by left clearance")
            y = limited
        elif y < 0:
            limited = self._scale_for_distance(y, sectors.get("right"), 0.27, 0.58)
            if limited != y:
                reasons.append("right strafe limited by right clearance")
            y = limited

        if yaw > 0 and sectors.get("left") is not None and sectors["left"] < 0.25:
            yaw = 0.0
            reasons.append("left turn limited by left clearance")
        elif yaw < 0 and sectors.get("right") is not None and sectors["right"] < 0.25:
            yaw = 0.0
            reasons.append("right turn limited by right clearance")
        return (x, y, yaw), ", ".join(reasons)

    @staticmethod
    def _clearance(sectors: dict, name: str) -> float:
        value = sectors.get(name)
        return float(value) if value is not None else AUTO_OPEN_DISTANCE

    def _auto_vector(
        self, lidar: dict, now: float
    ) -> tuple[tuple[float, float, float], str, str]:
        if not lidar["fresh"]:
            return (0.0, 0.0, 0.0), "Auto waiting: LiDAR data is stale", "waiting"
        scan_points = int(lidar.get("scan_points", 0) or 0)
        if scan_points < AUTO_MIN_SCAN_POINTS:
            return (
                (0.0, 0.0, 0.0),
                f"Auto waiting: only {scan_points}/{AUTO_MIN_SCAN_POINTS} recent scan points",
                "waiting",
            )
        sectors = lidar["sectors"]
        front = sectors.get("front")
        front_clearance = self._clearance(sectors, "front")
        front_left_clearance = self._clearance(sectors, "front_left")
        front_right_clearance = self._clearance(sectors, "front_right")
        left_clearance = self._clearance(sectors, "left")
        right_clearance = self._clearance(sectors, "right")
        rear_clearance = self._clearance(sectors, "rear")

        # Turning needs both the diagonal and side arc to be clear.  A close
        # return on the opposite side has no authority over this maneuver.
        left_route = min(front_left_clearance, left_clearance)
        right_route = min(front_right_clearance, right_clearance)

        if self._auto_reverse_until > now:
            if rear_clearance > AUTO_REVERSE_STOP_DISTANCE:
                return (
                    (-0.040, 0.0, 0.0),
                    f"Auto rear escape; rear {rear_clearance:.2f} m",
                    "reverse-escape",
                )
            self._auto_reverse_until = 0.0

        if (
            self._auto_turn
            and now < self._auto_turn_until
            and front_clearance < AUTO_TURN_RELEASE_DISTANCE
        ):
            direction = "left" if self._auto_turn > 0 else "right"
            route_clearance = left_route if self._auto_turn > 0 else right_route
            if route_clearance > AUTO_TRAPPED_DISTANCE:
                return (
                    (0.0, 0.0, 0.26 * self._auto_turn),
                    f"Auto turning {direction}; route {route_clearance:.2f} m",
                    f"turn-{direction}",
                )
            self._auto_turn = 0

        if front_clearance <= AUTO_TURN_DISTANCE:
            best_side = max(left_route, right_route)
            if best_side >= AUTO_SIDE_ROUTE_DISTANCE:
                self._auto_turn = 1 if left_route >= right_route else -1
                self._auto_turn_until = now + 0.70
                direction = "left" if self._auto_turn > 0 else "right"
                return (
                    (0.0, 0.0, 0.26 * self._auto_turn),
                    f"Front blocked; Auto selected {direction} route ({best_side:.2f} m)",
                    f"turn-{direction}",
                )
            if rear_clearance >= AUTO_REAR_ESCAPE_DISTANCE:
                self._auto_turn = 0
                self._auto_reverse_until = now + 0.90
                return (
                    (-0.040, 0.0, 0.0),
                    f"Front and sides blocked; Auto selected rear ({rear_clearance:.2f} m)",
                    "reverse-escape",
                )
            self._auto_turn = 0
            return (
                (0.0, 0.0, 0.0),
                "Auto stopped: front, turn paths, and rear are blocked",
                "trapped",
            )

        if front_clearance < AUTO_CAUTION_DISTANCE:
            preferred = 1 if left_route >= right_route else -1
            ratio = (front_clearance - AUTO_TURN_DISTANCE) / (
                AUTO_CAUTION_DISTANCE - AUTO_TURN_DISTANCE
            )
            speed = 0.025 + 0.075 * max(0.0, min(1.0, ratio))
            direction = "left" if preferred > 0 else "right"
            route_clearance = left_route if preferred > 0 else right_route
            yaw = 0.12 * preferred if route_clearance > AUTO_TRAPPED_DISTANCE else 0.0
            return (
                (speed, 0.0, yaw),
                (
                    f"Auto curving {direction}; front {front_clearance:.2f} m"
                    if yaw
                    else f"Auto moving straight cautiously; front {front_clearance:.2f} m"
                ),
                "cautious",
            )

        self._auto_turn = 0
        self._auto_reverse_until = 0.0
        description = "open" if front is None else f"{front_clearance:.2f} m"
        return (0.10, 0.0, 0.0), f"Auto forward at full speed; front {description}", "forward"

    def _rising(self, buttons: dict, name: str) -> bool:
        return bool(buttons.get(name)) and not bool(self._previous_buttons.get(name))

    def _handle_gamepad_buttons(self, gamepad: dict, now: float) -> None:
        buttons = gamepad["buttons"] if gamepad["online"] else {}
        if self._rising(buttons, "l3"):
            self.emergency_stop("Remote L3 emergency stop")
        if self._rising(buttons, "select"):
            with self._lock:
                new_mode = "manual" if self._mode == "guard" else "guard"
            self.set_mode(new_mode)
        if self._rising(buttons, "start"):
            with self._lock:
                ready = self._robot_connected and self._robot_armed
            if ready:
                try:
                    self._home_fn()
                except Exception:
                    pass

        pose_changed = False
        with self._lock:
            if now - self._last_pose_send >= 0.10:
                if buttons.get("cross") and self._pose["z"] < -0.10:
                    self._pose["z"] = min(-0.10, self._pose["z"] + 0.002)
                    pose_changed = True
                if buttons.get("triangle") and self._pose["z"] > -0.15:
                    self._pose["z"] = max(-0.15, self._pose["z"] - 0.002)
                    pose_changed = True
                if buttons.get("l1") and self._pose["pitch"] < 0.35:
                    self._pose["pitch"] = min(0.35, self._pose["pitch"] + 0.02)
                    pose_changed = True
                if buttons.get("l2") and self._pose["pitch"] > -0.35:
                    self._pose["pitch"] = max(-0.35, self._pose["pitch"] - 0.02)
                    pose_changed = True
                if buttons.get("r1") and self._pose["roll"] > -0.17:
                    self._pose["roll"] = max(-0.17, self._pose["roll"] - 0.02)
                    pose_changed = True
                if buttons.get("r2") and self._pose["roll"] < 0.17:
                    self._pose["roll"] = min(0.17, self._pose["roll"] + 0.02)
                    pose_changed = True
                pose = dict(self._pose)
                ready = self._robot_connected and self._robot_armed
                if pose_changed:
                    self._last_pose_send = now
            else:
                pose = dict(self._pose)
                ready = False
        if pose_changed and ready:
            try:
                self._pose_fn(
                    pose["x"], pose["y"], pose["z"], pose["x_shift"],
                    pose["roll"], pose["pitch"], 0.10,
                )
            except Exception:
                pass
        self._previous_buttons = dict(buttons)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            gamepad = self.gamepad.snapshot()
            lidar = self.lidar.snapshot()
            self._handle_gamepad_buttons(gamepad, started)
            remote = self._remote_vector(gamepad)

            with self._lock:
                connected = self._robot_connected
                armed = self._robot_armed
                mode = self._mode
                web = self._web_vector if started <= self._web_expires else (0.0, 0.0, 0.0)
                previous_command = self._commanded
                previous_source = self._input_source
                stop_requested = self._stop_requested
                stop_reason = self._stop_reason
                self._stop_requested = False

            if stop_requested:
                try:
                    self._stop_fn()
                except Exception:
                    pass
                with self._lock:
                    self._was_moving = False
                    self._commanded = (0.0, 0.0, 0.0)
                    self._blocked_reason = stop_reason
                previous_command = (0.0, 0.0, 0.0)

            remote_override = self._auto_override_active(gamepad) if mode == "auto" else _nonzero(remote)

            if remote_override:
                source = "gamepad"
                desired = remote
                if mode == "auto":
                    with self._lock:
                        self._mode = "guard"
                        mode = "guard"
                        self._auto_turn = 0
                        self._auto_reverse_until = 0.0
            elif _nonzero(web):
                source = "web"
                desired = web
                if mode == "auto":
                    with self._lock:
                        self._mode = "guard"
                        mode = "guard"
                        self._auto_turn = 0
                        self._auto_reverse_until = 0.0
            elif mode == "auto":
                source = "auto"
                desired = (0.0, 0.0, 0.0)
            else:
                source = "none"
                desired = (0.0, 0.0, 0.0)

            reason = ""
            if not connected or not armed:
                command = (0.0, 0.0, 0.0)
                reason = "Robot is offline or software-locked"
                with self._lock:
                    self._auto_state = "offline" if mode == "auto" else "idle"
            elif mode == "auto":
                if not gamepad["online"]:
                    command = (0.0, 0.0, 0.0)
                    reason = "Auto stopped: remote override is unavailable"
                    with self._lock:
                        self._auto_state = "remote-offline"
                else:
                    command, reason, auto_state = self._auto_vector(lidar, started)
                    with self._lock:
                        self._auto_state = auto_state
            elif mode == "guard":
                command, reason = self._guard_vector(desired, lidar)
                with self._lock:
                    self._auto_state = "idle"
            else:
                command = desired
                reason = "Manual mode bypasses LiDAR" if _nonzero(command) else ""
                with self._lock:
                    self._auto_state = "idle"

            target = command
            target_moving = _nonzero(target)
            send_due = started - self._last_send >= 0.22
            gamepad_lost = previous_source == "gamepad" and not gamepad["online"]
            smooth_release = (
                not target_moving
                and _nonzero(previous_command)
                and connected
                and armed
                and not reason
                and mode in ("manual", "guard")
                and not gamepad_lost
            )
            actual = previous_command
            try:
                if send_due and (target_moving or smooth_release):
                    next_command = _slew_vector(previous_command, target)
                    if _nonzero(next_command) and self._move_fn(*next_command):
                        self._last_send = started
                        self._was_moving = True
                        actual = next_command
                    elif not _nonzero(next_command):
                        if self._was_moving:
                            self._stop_fn()
                        self._was_moving = False
                        actual = (0.0, 0.0, 0.0)
                    else:
                        actual = (0.0, 0.0, 0.0)
                        reason = "Controller rejected movement"
                        self._was_moving = False
                elif not target_moving and not smooth_release and self._was_moving:
                    self._stop_fn()
                    self._was_moving = False
                    actual = (0.0, 0.0, 0.0)
            except Exception as exc:
                actual = (0.0, 0.0, 0.0)
                reason = f"Motion bridge error: {exc}"
                self._was_moving = False

            with self._lock:
                self._input_source = source
                self._desired = desired
                self._commanded = actual
                self._blocked_reason = reason

            elapsed = time.monotonic() - started
            self._stop_event.wait(max(0.01, 0.05 - elapsed))
