"""
autonomy.py - Robodog Pro v8.4.6 motion ownership and autonomous behavior.

Autonomous wander (v8.4) uses the Modulino IMU (no camera required):
  * Walk forward, then turn; TURN DIRECTION ALTERNATES every recovery so the
    dog never loops into a corner.
  * Obstacle / "walked into a wall" detection is HYBRID:
      - If the Edge Impulse "Vibration Anomaly Detection" brick is available
        and returns an anomaly score, that score is used.
      - Otherwise it falls back to an on-device z-score of the dog's own
        learned walking-vibration envelope (works today, no training).
  * On an anomaly: stop, back up, turn ~180 (alternating side), look around,
    relearn the baseline, continue.

Feedback (buzzer + built-in RGB LED) is handled entirely on the MCU: every
mode/event calls sound_event(id) over the Bridge, and the sketch drives both
the buzzer pattern AND the RGB status LED for that id. No led_control.py and no
hw_bridge.py changes are required.
    18 = manual (green)   19 = autonomous (blue)   20 = wall hit (red blink)
     6 = walkaround (blue) 12 = impact (red blink)  14 = low battery (purple)
"""

import collections
import os
import statistics
import threading
import time

import state as S
import hw_bridge as HW


# ============================================================================
# Command / action maps
# ============================================================================
COMMAND_MAP = {
    "forward": "dog_forward", "backward": "dog_backward",
    "left": "dog_left", "right": "dog_right",
    "turn_left": "dog_turn_left", "turn_right": "dog_turn_right",
    "stop": "dog_stop", "battery": "dog_battery",
    "gait_trot": "dog_gait_trot", "gait_amble": "dog_gait_amble", "gait_high": "dog_gait_high",
    "mark_time_on": "dog_mark_time_25", "mark_time_off": "dog_mark_time_off",
    "motors_on": "dog_load_motors", "motors_off": "dog_unload_motors",
    "attitude_reset": "dog_attitude_reset", "translate_reset": "dog_translate_reset",
    "perform_on": "dog_perform_on", "perform_off": "dog_perform_off",
    "pace_normal": "dog_pace_normal", "pace_slow": "dog_pace_slow", "pace_high": "dog_pace_high",
    "imu_on": "dog_imu_on", "imu_off": "dog_imu_off",
}

NAMED_ACTIONS = {
    "lie_down": 1, "stand_up": 2, "creep_fwd": 3, "circle": 4, "squat": 5,
    "roll": 6, "stretch": 7, "snake": 8, "pray": 9, "look_food": 10,
    "handshake": 11, "nod": 12, "pushup": 13, "look_around": 14,
    "wave": 15, "dance": 16, "pee": 17, "sit_beg": 18,
    "sway_roll": 19, "sway_pitch": 20, "sway_yaw": 21,
    "chicken_head": 22, "twist": 23, "reset": 24,
}


_COMMAND_LOCK = threading.Lock()
_LAST_COMMAND_SENT = None
_LAST_COMMAND_SENT_AT = 0.0
COMMAND_RESEND_SEC = 0.40


def send_command(cmd, force=False, announce=True):
    """Send one motion command with bounded periodic refresh and no Bridge flood."""
    global _LAST_COMMAND_SENT, _LAST_COMMAND_SENT_AT
    cmd = str(cmd).strip()
    now = time.monotonic()
    with _COMMAND_LOCK:
        if (not force and cmd == _LAST_COMMAND_SENT and
                now - _LAST_COMMAND_SENT_AT < COMMAND_RESEND_SEC):
            return {"ok": True, "cmd": cmd, "deduplicated": True}
        _LAST_COMMAND_SENT = cmd
        _LAST_COMMAND_SENT_AT = now
    S.STATE["last_command"] = cmd
    if cmd not in COMMAND_MAP:
        S.robot_say(f"Unknown command: {cmd}")
        return {"ok": False, "error": f"Unknown command: {cmd}"}
    result = HW.bridge_call(COMMAND_MAP[cmd])
    if cmd == "battery":
        HW.matrix_enqueue("battery")
        S.robot_say("Battery query sent.")
    elif announce:
        S.robot_say(f"Command: {cmd}")
    return {
        "ok": result is not None,
        "cmd": cmd,
        "bridge": COMMAND_MAP[cmd],
        "result": result,
        "deduplicated": False,
    }


def send_action(action_id):
    try:
        action_id = int(action_id)
    except Exception:
        return {"ok": False, "error": "Invalid action id"}
    action_id = max(1, min(255, action_id))
    S.STATE["last_action"] = action_id
    result = HW.bridge_call("dog_action", action_id)
    S.robot_say(f"Running action {action_id}")
    return {"ok": result is not None, "action": action_id, "result": result}


def send_action_named(name):
    name = str(name).strip().lower()
    aid = NAMED_ACTIONS.get(name)
    if not aid:
        return {"ok": False, "error": f"unknown action '{name}'"}
    return send_action(aid)


# ============================================================================
# Vision steering intent (kept for when a camera IS present)
# ============================================================================
def steering_intent_from_center_x(cx, image_width, left_threshold=0.35,
                                  right_threshold=0.65):
    """Create a conservative horizontal steering intent for a visible target."""
    rel = float(cx) / max(float(image_width), 1.0)
    if rel < left_threshold:
        return {"linear_x": 0.0, "angular_z": +0.5, "command": "turn_left"}
    if rel > right_threshold:
        return {"linear_x": 0.0, "angular_z": -0.5, "command": "turn_right"}
    return {"linear_x": 0.3, "angular_z": 0.0, "command": "forward"}


def apply_intent(intent):
    cmd = intent.get("command") if isinstance(intent, dict) else None
    if cmd:
        return send_command(cmd, announce=False)
    return {"ok": False, "error": "intent contains no command"}


_last_vision_seen_at = 0.0
_last_vision_lost_stop_at = 0.0
VISION_LOST_STOP_RESEND_SEC = 1.0


def live_autonomy_step(target_det, image_width, image_height=None):
    """Follow a visible target and stop safely when the target is absent."""
    global _last_vision_seen_at, _last_vision_lost_stop_at
    if S.STATE.get("mode") != "autonomous_vision":
        return

    st = S.STATE.setdefault("autonomy", {})
    now = time.monotonic()
    st.update(active=True, owner="vision")

    if not target_det:
        st.update(phase="target_lost", reason="vision target unavailable")
        if now - _last_vision_lost_stop_at >= VISION_LOST_STOP_RESEND_SEC:
            send_command("stop", force=True, announce=False)
            _last_vision_lost_stop_at = now
        return

    from perception import detection_center_x
    _last_vision_seen_at = now
    cx = detection_center_x(target_det, image_width)
    intent = steering_intent_from_center_x(cx, image_width)
    st.update(
        phase="tracking",
        reason="target visible",
        last_seen_at=time.time(),
        vision_command=intent["command"],
    )
    apply_intent(intent)


def one_shot_autonomy_apply(cx, image_width):
    if S.STATE.get("mode") != "autonomous_vision":
        return
    apply_intent(steering_intent_from_center_x(cx, image_width))


# ============================================================================
# Battery hysteresis -> mood
# ============================================================================
def apply_battery_mood(battery_pct):
    try:
        b = int(battery_pct)
    except Exception:
        return
    if not (0 <= b <= 100):
        return
    if b < S.LOW_BATTERY_PCT and not S.MOOD["low_battery_latched"]:
        HW.set_mood("low_energy")
        HW.bridge_call_quiet("sound_event", 14)   # low battery -> purple LED + tone
        S.MOOD["low_battery_latched"] = True
    elif b >= S.LOW_BATTERY_RECOVERY_PCT and S.MOOD["low_battery_latched"]:
        S.MOOD["low_battery_latched"] = False
        HW.set_mood("normal")


# ============================================================================
# Mode transitions (buzzer voice + LED handled on MCU via sound_event)
# ============================================================================
def set_mode(mode):
    mode = str(mode).strip().lower()
    aliases = {"autonomous": "autonomous_imu", "auto": "autonomous_imu", "vision": "autonomous_vision", "imu": "autonomous_imu"}
    mode = aliases.get(mode, mode)
    if mode not in ("manual", "autonomous_imu", "autonomous_vision"):
        mode = "manual"
    S.STATE["mode"] = mode
    owner = "manual" if mode == "manual" else ("vision" if mode == "autonomous_vision" else "imu")
    autonomy = S.STATE.setdefault("autonomy", {})
    autonomy.update(
        owner=owner,
        requested_owner=owner,
        active=(mode != "manual"),
        phase="idle" if mode == "manual" else "starting",
        reason=None,
    )
    if mode == "manual":
        send_command("stop", force=True, announce=False)
        HW.bridge_call_quiet("sound_event", 18)   # manual: calm tone + green LED
        HW.matrix_enqueue("scroll", "MANUAL MODE", 35)
        HW.set_mood("normal")
        S.robot_say("Manual mode.")
    else:
        _reset_anomaly_baseline()
        HW.bridge_call_quiet("sound_event", 19)   # autonomous: rising tone + blue LED
        HW.matrix_enqueue("scroll", "AUTO VISION" if mode == "autonomous_vision" else "AUTO IMU MODE", 35)
        HW.set_mood("happy")
        S.robot_say("Autonomous vision follow mode." if mode == "autonomous_vision" else "Autonomous IMU mode: wandering; anomaly detection armed.")
    return {"ok": True, "mode": S.STATE["mode"]}


# ============================================================================
# Dog-like walkaround sequence (non-blocking, cancellable)
# ============================================================================
_SEQUENCE_LOCK = threading.Lock()
_SEQUENCE_CANCEL = threading.Event()
_SEQUENCE_THREAD = None


def _sequence_sleep(seconds):
    return not _SEQUENCE_CANCEL.wait(max(0.0, float(seconds)))


def _walkaround_worker():
    global _SEQUENCE_THREAD
    try:
        S.STATE["sequence"].update(running=True, name="walkaround", step="start", error=None)
        HW.matrix_enqueue("scroll", "DOG MODE", 30)
        HW.bridge_call_quiet("sound_event", 6)   # walkaround tone + blue LED
        S.robot_say("Dog-like walkaround sequence started.")
        steps = [
            ("look_around", lambda: send_action_named("look_around"), 3.0),
            ("look_food",   lambda: send_action_named("look_food"),   3.0),
            ("sit_beg",     lambda: send_action_named("sit_beg"),     3.5),
            ("circle",      lambda: send_action_named("circle"),      4.0),
            ("forward",     lambda: send_command("forward"),          1.2),
            ("stop",        lambda: send_command("stop"),             0.5),
            ("wave",        lambda: send_action_named("wave"),        2.5),
        ]
        for label, fn, hold in steps:
            if _SEQUENCE_CANCEL.is_set():
                break
            S.STATE["sequence"]["step"] = label
            fn()
            if not _sequence_sleep(hold):
                break
    except Exception as exc:
        S.STATE["sequence"]["error"] = str(exc)
    finally:
        send_command("stop")
        cancelled = _SEQUENCE_CANCEL.is_set()
        S.STATE["sequence"].update(running=False, step="cancelled" if cancelled else "complete")
        HW.matrix_enqueue("scroll", "STOP" if cancelled else "GOOD DOG", 35)

        # Event 18 always releases the MCU walkaround face-ownership lock and
        # restores the normal manual-ready policy. A successful sequence then
        # receives the short greeting cue without leaving the face locked.
        HW.bridge_call_quiet("sound_event", 18)
        if not cancelled:
            time.sleep(0.15)
            HW.bridge_call_quiet("sound_event", 24)

        with _SEQUENCE_LOCK:
            _SEQUENCE_THREAD = None


def start_walkaround_sequence():
    global _SEQUENCE_THREAD
    with _SEQUENCE_LOCK:
        if _SEQUENCE_THREAD and _SEQUENCE_THREAD.is_alive():
            return {"ok": True, "started": False, "reason": "already running", **S.STATE["sequence"]}
        _SEQUENCE_CANCEL.clear()
        _SEQUENCE_THREAD = threading.Thread(target=_walkaround_worker,
                                            name="walkaround-sequence", daemon=True)
        _SEQUENCE_THREAD.start()
    return {"ok": True, "started": True, **S.STATE["sequence"]}


def stop_walkaround_sequence():
    _SEQUENCE_CANCEL.set()
    send_command("stop")
    return {"ok": True, "cancel_requested": True, **S.STATE["sequence"]}


# ============================================================================
# v8.4 HYBRID anomaly detection: Edge Impulse brick -> z-score fallback
# ============================================================================
ANOMALY_WINDOW   = 25
ANOMALY_Z        = 4.0
ANOMALY_ABS_MIN  = 0.80
ANOMALY_WARMUP   = 12
EI_ANOMALY_LIMIT = 0.60   # brick anomaly score above this = obstacle (tune per model)

_vib_window = collections.deque(maxlen=ANOMALY_WINDOW)

# Try to load the Edge Impulse Vibration Anomaly Detection brick. Confirm the
# exact import/class from the brick's "API documentation" tab; adjust below if
# needed. If unavailable, we transparently use the z-score path.
_EI = None
EI_ENABLED = os.environ.get("ROBODOG_ENABLE_EI_ANOMALY", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
try:
    if not EI_ENABLED:
        raise RuntimeError("Edge Impulse anomaly path disabled until a DOGZILLA-trained model is deployed")
    from arduino.app_bricks.vibration_anomaly_detection import VibrationAnomalyDetection  # type: ignore
    _EI = VibrationAnomalyDetection()
    print("[anomaly] Edge Impulse Vibration Anomaly brick loaded.", flush=True)
except Exception:
    try:
        if not EI_ENABLED:
            raise RuntimeError("generic Edge Impulse path disabled")
        from arduino.app_bricks.anomaly_detection import AnomalyDetection  # type: ignore
        _EI = AnomalyDetection()
        print("[anomaly] Edge Impulse Anomaly brick loaded (generic).", flush=True)
    except Exception:
        _EI = None
        print("[anomaly] using validated on-device z-score; EI requires ROBODOG_ENABLE_EI_ANOMALY=1 and a DOGZILLA-trained model.", flush=True)


def _reset_anomaly_baseline():
    global _anomaly_confirm_count
    _vib_window.clear()
    _anomaly_confirm_count = 0


def _ei_anomaly_score(x, y, z):
    """
    OPTIONAL Edge Impulse path. Returns a float anomaly score, or None.

    NOTE: the stock 'fan' model expects accX/accY/accZ at ~100 Hz and knows
    'fan on/off'. Our Bridge IMU is ~5 Hz. For meaningful dog collision
    detection, TRAIN THE BRICK ON YOUR DOG (walking=nominal, bumps=anomaly)
    in Edge Impulse, then redeploy. Until then z-score is the reliable path.

    Adapt this single call to match the brick's API documentation tab, e.g.:
        return _EI.classify([x, y, z]).get("anomaly")
    """
    if _EI is None:
        return None
    try:
        if hasattr(_EI, "classify"):
            r = _EI.classify([x, y, z])
        elif hasattr(_EI, "run"):
            r = _EI.run([x, y, z])
        elif hasattr(_EI, "predict"):
            r = _EI.predict([x, y, z])
        else:
            return None
        if isinstance(r, dict):
            for k in ("anomaly", "anomaly_score", "score"):
                if k in r:
                    return float(r[k])
        return float(r)
    except Exception:
        return None


_anomaly_confirm_count = 0
ANOMALY_CONFIRM_SAMPLES = 3
ANOMALY_COOLDOWN_SEC = 4.0
_last_anomaly_at = 0.0


def _detect_anomaly(x, y, z, vibration):
    """Return a confirmed anomaly tuple without learning the candidate spike."""
    global _anomaly_confirm_count, _last_anomaly_at
    st = S.STATE.setdefault("autonomy", {})
    now = time.monotonic()

    ei = _ei_anomaly_score(x, y, z) if EI_ENABLED else None
    if ei is not None:
        st["ei_score"] = round(ei, 3)
        st["anom_src"] = "edge_impulse"
        candidate = ei >= EI_ANOMALY_LIMIT
        metric = ei
        source = "edge_impulse"
    else:
        st["anom_src"] = "zscore"
        if len(_vib_window) < ANOMALY_WARMUP:
            _vib_window.append(vibration)
            st["baseline_samples"] = len(_vib_window)
            return False, 0.0, "zscore"

        mean = statistics.mean(_vib_window)
        std = max(statistics.pstdev(_vib_window), 0.02)
        zc = (vibration - mean) / std
        st.update(z=round(zc, 2), vib_mean=round(mean, 3), vib_std=round(std, 3))
        candidate = zc >= ANOMALY_Z and vibration >= ANOMALY_ABS_MIN
        metric = zc
        source = "zscore"

        # Learn nominal movement only. Never contaminate the baseline with a
        # candidate collision spike.
        if not candidate:
            _vib_window.append(vibration)

    if candidate:
        _anomaly_confirm_count += 1
    else:
        _anomaly_confirm_count = max(0, _anomaly_confirm_count - 1)
    st["anomaly_confirm_count"] = _anomaly_confirm_count

    confirmed = (
        _anomaly_confirm_count >= ANOMALY_CONFIRM_SAMPLES
        and now - _last_anomaly_at >= ANOMALY_COOLDOWN_SEC
    )
    if confirmed:
        _last_anomaly_at = now
        _anomaly_confirm_count = 0
        st["anomaly_confirm_count"] = 0
    return confirmed, metric, source


# Alternating turn direction so recovery never loops into a corner.
_next_turn_dir = "turn_right"


def _flip_turn():
    global _next_turn_dir
    d = _next_turn_dir
    _next_turn_dir = "turn_left" if d == "turn_right" else "turn_right"
    return d


def _recover_from_wall(metric, source):
    st = S.STATE["autonomy"]
    st["phase"] = "recover_wall"
    st["impacts"] = int(st.get("impacts", 0)) + 1
    st["reason"] = f"obstacle ({source} {metric:.2f})"
    HW.set_mood("alert")
    HW.bridge_call_quiet("sound_event", 20)   # wall hit: tone + red LED blink (MCU)
    S.robot_say(f"Obstacle detected ({source} {metric:.2f}). Backing up and turning around.")
    send_command("stop"); time.sleep(0.2)
    send_command("backward"); time.sleep(S.AUTONOMY_RECOVER_BACK_SEC)
    send_command("stop"); time.sleep(0.1)
    turn = _flip_turn()                       # ALTERNATES each recovery
    send_command(turn); time.sleep(S.AUTONOMY_TURN_SEC * 2.0)  # ~180 degrees
    send_command("stop"); time.sleep(0.1)
    send_action_named("look_around"); time.sleep(2.0)
    _reset_anomaly_baseline()
    HW.set_mood("happy")
    HW.bridge_call_quiet("sound_event", 19)   # back to autonomous -> blue LED


def _recover_from_tilt():
    st = S.STATE["autonomy"]
    st["phase"] = "recover_tilt"
    HW.set_mood("alert")
    HW.bridge_call_quiet("sound_event", 12)   # impact tone + red LED blink
    send_command("stop"); time.sleep(0.2)
    send_action_named("stand_up"); time.sleep(2.0)
    HW.set_mood("happy")
    HW.bridge_call_quiet("sound_event", 19)


def _read_imu():
    m = S.MOTION
    return (HW.safe_float(m.get("x")), HW.safe_float(m.get("y")),
            HW.safe_float(m.get("z")), HW.safe_float(m.get("vibration")))


# ============================================================================
# Orientation-neutral tilt confirmation
# ============================================================================
_TILT_BASELINE_ALPHA = 0.02
_TILT_DELTA_G = 0.55


def _tilt_delta_g(x, y, z, baseline):
    """Return vector distance from the learned mounted resting orientation."""
    if baseline is None:
        return 0.0
    bx, by, bz = baseline
    return ((x - bx) ** 2 + (y - by) ** 2 + (z - bz) ** 2) ** 0.5


# ============================================================================
# v8.4.6 IMU-driven autonomous wander loop
# ============================================================================
def autonomous_worker():
    """Single-owner IMU wander worker. Vision control remains callback-driven."""
    phase = "forward"
    phase_started = time.monotonic()
    last_sniff = time.monotonic()
    tilt_unsafe_count = 0
    tilt_safe_count = 0
    tilt_baseline = None
    tilt_baseline_samples = 0

    while True:
        S.heartbeat("autonomy")
        try:
            mode = S.STATE.get("mode", "manual")
            st = S.STATE.setdefault("autonomy", {})

            # Critical ownership rule: this worker moves the robot only in
            # autonomous_imu. autonomous_vision is owned by live_autonomy_step.
            if mode != "autonomous_imu":
                if mode == "manual":
                    st.update(active=False, owner="manual", phase="idle")
                elif mode == "autonomous_vision":
                    st.update(active=True, owner="vision")
                time.sleep(S.AUTONOMY_TICK_SEC)
                phase = "forward"
                phase_started = time.monotonic()
                tilt_unsafe_count = 0
                tilt_safe_count = 0
                tilt_baseline = None
                tilt_baseline_samples = 0
                continue

            st.update(active=True, owner="imu")
            x, y, z, vib = _read_imu()
            st.update(
                vibration=round(vib, 3),
                x=round(x, 3),
                y=round(y, 3),
                z_axis=round(z, 3),
            )

            # Learn the sensor's actual mounted gravity vector. Absolute X/Y
            # thresholds are invalid when the Modulino is mounted sideways and
            # were treating the normal y ~= 0.93 g reading as a permanent tip.
            if tilt_baseline is None:
                tilt_baseline = (x, y, z)
                tilt_baseline_samples = 1
            elif tilt_baseline_samples < 20:
                n = float(tilt_baseline_samples)
                bx, by, bz = tilt_baseline
                tilt_baseline = (
                    (bx * n + x) / (n + 1.0),
                    (by * n + y) / (n + 1.0),
                    (bz * n + z) / (n + 1.0),
                )
                tilt_baseline_samples += 1

            tilt_delta = _tilt_delta_g(x, y, z, tilt_baseline)
            unsafe_tilt = tilt_baseline_samples >= 8 and tilt_delta >= _TILT_DELTA_G
            if unsafe_tilt:
                tilt_unsafe_count += 1
                tilt_safe_count = 0
            else:
                tilt_safe_count += 1
                if tilt_safe_count >= int(getattr(S, "AUTONOMY_SAFETY_CLEAR_SAMPLES", 8)):
                    tilt_unsafe_count = 0
                # Slowly adapt only while safe so normal mounting drift is learned
                # but an actual tip is never absorbed into the baseline.
                if tilt_baseline_samples >= 20:
                    bx, by, bz = tilt_baseline
                    a = _TILT_BASELINE_ALPHA
                    tilt_baseline = (
                        bx * (1.0 - a) + x * a,
                        by * (1.0 - a) + y * a,
                        bz * (1.0 - a) + z * a,
                    )

            st["tilt_delta_g"] = round(tilt_delta, 3)
            st["tilt_baseline"] = [round(v, 3) for v in tilt_baseline]
            st["tilt_baseline_samples"] = tilt_baseline_samples
            st["tilt_confirm_count"] = tilt_unsafe_count
            if tilt_unsafe_count >= int(getattr(S, "AUTONOMY_SAFETY_CONFIRM_SAMPLES", 5)):
                _recover_from_tilt()
                tilt_unsafe_count = 0
                phase = "forward"
                phase_started = time.monotonic()
                continue

            is_anom, metric, src = _detect_anomaly(x, y, z, vib)
            if is_anom:
                _recover_from_wall(metric, src)
                phase = "forward"
                phase_started = time.monotonic()
                continue

            now = time.monotonic()
            if now - last_sniff >= S.AUTONOMY_SNIFF_EVERY_SEC:
                last_sniff = now
                st.update(phase="sniff", reason="scheduled environmental check")
                send_command("stop", force=True, announce=False)
                send_action_named("look_around")
                time.sleep(1.5)
                phase = "forward"
                phase_started = time.monotonic()
                continue

            if phase == "forward":
                st.update(phase="forward", reason="scheduled wander")
                send_command("forward", announce=False)
                if now - phase_started >= S.AUTONOMY_FORWARD_SEC:
                    phase = "turn"
                    phase_started = now
                    st["_turn"] = _flip_turn()
            else:
                turn = st.get("_turn", "turn_left")
                st.update(phase="turn", reason=turn)
                send_command(turn, announce=False)
                if now - phase_started >= S.AUTONOMY_TURN_SEC:
                    phase = "forward"
                    phase_started = now

            time.sleep(S.AUTONOMY_TICK_SEC)

        except Exception as exc:
            st = S.STATE.setdefault("autonomy", {})
            st.update(active=False, phase="error", reason=f"worker error: {exc}")
            print(f"[autonomy] error: {exc}", flush=True)
            try:
                send_command("stop", force=True, announce=False)
                HW.bridge_call_quiet("sound_event", 29)
            except Exception:
                pass
            time.sleep(1.0)

