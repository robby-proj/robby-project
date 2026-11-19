#!/usr/bin/env python3
"""Robodog Pro v8.4.8 package consistency and zero-persistence self-check."""
from pathlib import Path
import ast
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
BUILD_ID = 'robodog-2026.08.19-v8.4.6-release-candidate'
errors = []
passes = []

required = [
    "app.yaml", "python/main.py", "python/state.py", "python/hw_bridge.py",
    "python/autonomy.py", "python/ui_api.py", "python/perception.py",
    "python/input_controller.py", "python/audio_feedback.py",
    "sketch/sketch.ino", "sketch/sketch.yaml", "host/dogzilla-ap.sh",
    "assets/app.js", "assets/index.html", "assets/style.css",
]
for name in required:
    if not (ROOT / name).is_file():
        errors.append(f"missing {name}")

for path in sorted((ROOT / "python").glob("*.py")):
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        errors.append(f"{path.name}:{exc.lineno} {exc.msg}")
if not any("Syntax" in e or ".py:" in e for e in errors):
    passes.append("Python syntax")

def read(name):
    path = ROOT / name
    return path.read_text(encoding="utf-8") if path.is_file() else ""

main = read("python/main.py")
state = read("python/state.py")
ui = read("python/ui_api.py")
hw = read("python/hw_bridge.py")
autonomy = read("python/autonomy.py")
perception = read("python/perception.py")
controller = read("python/input_controller.py")
audio = read("python/audio_feedback.py")
sketch = read("sketch/sketch.ino")
app_js = read("assets/app.js")

for name, source in (("main.py", main), ("state.py", state)):
    if BUILD_ID not in source:
        errors.append(f"v8.4.6 build identity missing from {name}")

required_tokens = {
    "autonomy.py": ["autonomous_imu", "autonomous_vision", "target_lost", "ANOMALY_CONFIRM_SAMPLES"],
    "perception.py": ["set_person_detected", "request_mcu=False"],
    "input_controller.py": ["_autonomous_mode_active", "autonomous_imu", "autonomous_vision"],
    "hw_bridge.py": ["refresh_eyes_health", "refresh_face_matrix_health", "eyes_flush_error_count", "set_person_detected"],
    "ui_api.py": ["flush_error_count", "wifi_fallback", "mcu_event_ids"],
    "audio_feedback.py": ["action_confirm", "greeting", "motors_on", "error"],
    "sketch.ino": ["eyesFlushAttempt", "eyesRecoverBusIfNeeded", "eyesPersonFace", "eyes_bus_fault", "case 29:",
                   "ModulinoLEDMatrix faceMatrix", "FACE_FRAME_HAPPY", "FACE_FRAME_SAD",
                   "FACE_FRAME_SERIOUS", "faceNoteAggressiveAcceleration", "faceNoteBattery"],
}
sources = {"autonomy.py": autonomy, "perception.py": perception, "input_controller.py": controller,
           "hw_bridge.py": hw, "ui_api.py": ui, "audio_feedback.py": audio, "sketch.ino": sketch}
for name, tokens in required_tokens.items():
    for token in tokens:
        if token not in sources[name]:
            errors.append(f"missing {token} in {name}")

if "attachInterrupt(" in sketch or "digitalPinToInterrupt(" in sketch:
    errors.append("Zephyr-incompatible GPIO interrupt API remains in sketch")
if "Wire1.setClock(400000)" not in sketch:
    errors.append("Wire1 shared display/sensor 400 kHz policy missing")

for path in (ROOT / "python").glob("*.py"):
    if path.name == "self_check.py":
        continue
    source = path.read_text(encoding="utf-8")
    for forbidden in ("cv2.imwrite(", "tempfile.NamedTemporaryFile(", "tempfile.mkstemp("):
        if forbidden in source:
            errors.append(f"runtime persistence call {forbidden} in {path.name}")
if '"image_path"' in perception or "'image_path'" in perception:
    errors.append("image_path response field remains in perception.py")

if "robodog-pro-v8-4/" in ui + app_js:
    errors.append("stale v8.4 host path remains")
if "robodog-pro-v8-4-6" not in ui + app_js + state:
    errors.append("v8.4.6 host path missing")

if not errors:
    passes.extend([
        "v8.4.6 identity and ownership",
        "Zephyr-safe soft-off and OLED recovery",
        "person feedback and expanded buzzer map",
        "external Modulino face matrix policy",
        "RAM-only camera policy",
        "frontend/backend path consistency",
    ])

result = {
    "release": "8.4.6", "build_id": BUILD_ID,
    "passed": passes, "errors": errors,
    "status": "PASS" if not errors else "FAIL",
}
print(json.dumps(result, indent=2))
raise SystemExit(0 if not errors else 1)
