"""ROSPug UNO Q backend with USB gamepad, LD19 avoidance, and AP-ready Web UI."""

import json
import threading
import time

from arduino.app_utils import App, Bridge
from arduino.app_bricks.web_ui import WebUI

from autonomy import MotionSupervisor
from remote_io import GamepadReader, LD19Reader


ui = WebUI()
bridge_lock = threading.Lock()
last_state = {
    "connected": False,
    "armed": False,
    "bridge_error": "Waiting for the sketch...",
}

MOVE_VECTORS = {
    "forward": (0.09, 0.0, 0.0),
    "backward": (-0.09, 0.0, 0.0),
    "left": (0.0, 0.07, 0.0),
    "right": (0.0, -0.07, 0.0),
    "rotate_left": (0.0, 0.0, 0.30),
    "rotate_right": (0.0, 0.0, -0.30),
}

GAIT_NAMES = {
    1: "Trot",
    2: "Amble",
    3: "Walk",
    4: "Fast Trot",
    5: "Smooth Fast Trot",
}


def call_sketch(method, *args, timeout=2.0):
    """Serialize RPC calls so telemetry and motion cannot collide."""
    with bridge_lock:
        return Bridge.call(method, *args, timeout=timeout)


def bridge_move(x, y, yaw):
    return bool(call_sketch("rospug_move", x, y, yaw, timeout=1.0))


def bridge_stop():
    return bool(call_sketch("rospug_stop", timeout=1.0))


def bridge_home():
    return bool(call_sketch("rospug_home", timeout=1.0))


def bridge_pose(x, y, z, x_shift, roll, pitch, duration):
    return bool(
        call_sketch(
            "rospug_set_pose", x, y, z, x_shift, roll, pitch, duration, timeout=1.0
        )
    )


gamepad = GamepadReader()
lidar = LD19Reader()
supervisor = MotionSupervisor(
    gamepad=gamepad,
    lidar=lidar,
    move_fn=bridge_move,
    stop_fn=bridge_stop,
    home_fn=bridge_home,
    pose_fn=bridge_pose,
)
gamepad.start()
lidar.start()
supervisor.start()


def read_state():
    global last_state
    try:
        raw = call_sketch("rospug_status", timeout=1.0)
        state = json.loads(raw)
        state["bridge_error"] = ""
        supervisor.update_robot_state(state.get("connected", False), state.get("armed", False))
        state.update(supervisor.snapshot())
        last_state = state
    except Exception as exc:
        supervisor.update_robot_state(False, False)
        last_state = {
            **last_state,
            "connected": False,
            "armed": False,
            "bridge_error": f"Bridge error: {exc}",
            **supervisor.snapshot(),
        }
    return last_state


def send_state(client=None):
    state = read_state()
    if client is None:
        ui.send_message("rospug_state", state)
    else:
        ui.send_message("rospug_state", state, client)


def on_get_state(client, data):
    send_state(client)


def on_set_armed(client, data):
    enabled = bool((data or {}).get("enabled", False))
    confirmed = bool((data or {}).get("confirmed", False))
    if enabled and not confirmed:
        ui.send_message(
            "action_result",
            {"ok": False, "message": "Safety confirmation is required before initialization."},
            client,
        )
        return
    try:
        if not enabled:
            supervisor.emergency_stop("Software command lock enabled")
        result = call_sketch("rospug_set_armed", enabled, timeout=2.0)
        supervisor.update_robot_state(bool(last_state.get("connected")), bool(result))
        if enabled and not result:
            response = {"ok": False, "message": "Initialization blocked: no valid ROSPug frames."}
        else:
            response = {
                "ok": True,
                "message": (
                    "Initialization frame sent. The robot may stand now."
                    if result
                    else "Software command lock enabled. Servo power is still live."
                ),
            }
        ui.send_message("action_result", response, client)
    except Exception as exc:
        ui.send_message(
            "action_result",
            {"ok": False, "message": f"Arm command failed: {exc}"},
            client,
        )
    send_state(client)


def on_read_servo(client, data):
    try:
        servo_id = int((data or {}).get("id", 1))
        raw = call_sketch("rospug_read_servo", servo_id, timeout=2.0)
        ui.send_message("servo_result", json.loads(raw), client)
    except Exception as exc:
        ui.send_message(
            "servo_result",
            {"ok": False, "error": f"Servo read failed: {exc}"},
            client,
        )


def on_stop(client, data):
    supervisor.emergency_stop("Web STOP")
    try:
        result = bridge_stop()
        ui.send_message(
            "action_result",
            {
                "ok": result,
                "message": "Stop frame sent; mode returned to LiDAR Guard."
                if result
                else "Stop failed: ROSPug link is offline.",
            },
            client,
        )
    except Exception as exc:
        ui.send_message(
            "action_result",
            {"ok": False, "message": f"Stop command failed: {exc}"},
            client,
        )


def on_set_gait(client, data):
    try:
        preset = int((data or {}).get("preset", 1))
        if preset not in GAIT_NAMES:
            raise ValueError("Unknown gait preset")
        result = bool(call_sketch("rospug_set_gait", preset, timeout=1.0))
        ui.send_message(
            "action_result",
            {
                "ok": result,
                "action": "set_gait",
                "preset": preset,
                "message": f"{GAIT_NAMES[preset]} gait loaded."
                if result
                else "Gait blocked: initialize the online robot first.",
            },
            client,
        )
    except Exception as exc:
        ui.send_message(
            "action_result",
            {
                "ok": False,
                "action": "set_gait",
                "preset": (data or {}).get("preset"),
                "message": f"Gait command failed: {exc}",
            },
            client,
        )
    send_state(client)


def on_home(client, data):
    supervisor.clear_web_intent()
    try:
        result = bridge_home()
        ui.send_message(
            "action_result",
            {
                "ok": result,
                "message": "Hiwonder neutral pose sent."
                if result
                else "Home pose blocked: initialize the online robot first.",
            },
            client,
        )
    except Exception as exc:
        ui.send_message(
            "action_result",
            {"ok": False, "message": f"Home command failed: {exc}"},
            client,
        )


def on_motion(client, data):
    direction = str((data or {}).get("direction", ""))
    vector = MOVE_VECTORS.get(direction)
    if vector is None:
        ui.send_message(
            "motion_result",
            {"ok": False, "message": "Unknown movement direction."},
            client,
        )
        return
    if not last_state.get("connected") or not last_state.get("armed"):
        ui.send_message(
            "motion_result",
            {"ok": False, "message": "Movement blocked: robot is offline or software-locked."},
            client,
        )
        return
    supervisor.set_web_intent(vector)


def on_set_mode(client, data):
    mode = str((data or {}).get("mode", "guard"))
    confirmed = bool((data or {}).get("confirmed", False))
    ok, message = supervisor.set_mode(mode, confirmed)
    ui.send_message("action_result", {"ok": ok, "message": message}, client)
    send_state(client)


ui.on_message("get_state", on_get_state)
ui.on_message("set_armed", on_set_armed)
ui.on_message("read_servo", on_read_servo)
ui.on_message("stop", on_stop)
ui.on_message("set_gait", on_set_gait)
ui.on_message("home", on_home)
ui.on_message("motion", on_motion)
ui.on_message("set_mode", on_set_mode)


def loop():
    send_state()
    time.sleep(0.25)


App.run(user_loop=loop)
