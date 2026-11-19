"""
perception.py - Robodog Pro v8.4.7 RAM-only high-FPS camera and object detection.

Camera frames live only in CAMERA_STATE["frame_base64"] (replaced each UI
encode) and in state.LIVE_FRAME_RGB (single PIL image used by continuous
inference). One-shot detect annotates in RAM and returns base64 in the JSON
response. No capture directory is created, no image_path is exposed, and no
camera frame is written to disk.
"""

import base64
import io
import os
import time
from urllib.parse import unquote

from PIL import Image

# Silence OpenCV's GStreamer/V4L2 chatter BEFORE importing cv2. App Lab can
# expose non-capture /dev/video* nodes, so discovery explicitly uses V4L2.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")

try:
    import cv2
    try:
        cv2.setLogLevel(0)
    except Exception:
        pass
except Exception:
    cv2 = None

from arduino.app_bricks.object_detection import ObjectDetection

import state as S
import hw_bridge as HW

try:
    import audio_feedback as F
except Exception:
    F = None


object_detection = ObjectDetection()


# ============================================================================
# Camera performance policy
# ============================================================================
CAMERA_REQUEST_WIDTH = 640
CAMERA_REQUEST_HEIGHT = 480
CAMERA_REQUEST_FPS = 30
CAMERA_REQUEST_FOURCC = "MJPG"
CAMERA_BUFFER_SIZE = 1
CAMERA_OPEN_WARMUP_GRABS = 5

# Deliberately decouple capture, WebUI presentation, and AI inference rates
UI_TARGET_FPS = 10.0
UI_FRAME_INTERVAL = 1.0 / UI_TARGET_FPS
UI_JPEG_QUALITY = 65

DETECTION_WIDTH = 320
DETECTION_HEIGHT = 240

# A local fallback keeps live inference near 5 Hz even if an older state.py
# still has an aggressively small CONTINUOUS_DETECTION_INTERVAL value.
DEFAULT_DETECTION_INTERVAL = 0.20
MIN_DETECTION_INTERVAL = 0.10


# ============================================================================
# Helpers
# ============================================================================
def pil_to_jpeg_base64(img, quality=68):
    if img.mode != "RGB":
        img = img.convert("RGB")
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=quality)
    return base64.b64encode(b.getvalue()).decode("utf-8")


def normalize_label(label):
    return str(label).strip().lower().replace("_", " ")


def extract_detections(results):
    if not results:
        return []
    if isinstance(results, list):
        return results
    if isinstance(results, dict):
        for key in (
            "detections",
            "detection",
            "results",
            "boxes",
            "objects",
            "predictions",
        ):
            value = results.get(key)
            if isinstance(value, list):
                return value
        if any(k in results for k in ("label", "class", "score", "x", "x1", "left")):
            return [results]
    return []


def detection_label(det):
    if not isinstance(det, dict):
        return "unknown"
    for key in (
        "label",
        "class",
        "name",
        "category",
        "class_name",
        "class_label",
        "prediction",
    ):
        if key in det:
            return str(det[key])
    return "unknown"


def detection_score(det):
    if not isinstance(det, dict):
        return 0.0
    for key in ("score", "confidence", "value", "probability", "prob"):
        if key in det:
            try:
                return float(det[key])
            except Exception:
                return 0.0
    return 0.0


def detection_box_xyxy(det):
    if not isinstance(det, dict):
        return None
    try:
        if all(k in det for k in ("x", "y", "width", "height")):
            x1 = float(det["x"])
            y1 = float(det["y"])
            return (
                x1,
                y1,
                x1 + float(det["width"]),
                y1 + float(det["height"]),
            )
        if all(k in det for k in ("x1", "y1", "x2", "y2")):
            return (
                float(det["x1"]),
                float(det["y1"]),
                float(det["x2"]),
                float(det["y2"]),
            )
        if all(k in det for k in ("left", "top", "right", "bottom")):
            return (
                float(det["left"]),
                float(det["top"]),
                float(det["right"]),
                float(det["bottom"]),
            )
        if "bbox" in det:
            box = det["bbox"]
            if isinstance(box, (list, tuple)) and len(box) == 4:
                return (
                    float(box[0]),
                    float(box[1]),
                    float(box[0]) + float(box[2]),
                    float(box[1]) + float(box[3]),
                )
    except Exception:
        return None
    return None


def detection_center_x(det, image_width):
    box = detection_box_xyxy(det)
    if box:
        x1, _, x2, _ = box
        return (x1 + x2) / 2.0
    return image_width / 2.0


def list_video_devices():
    try:
        return sorted([d for d in os.listdir("/dev") if d.startswith("video")])
    except Exception:
        return []


def _bilinear_resampling():
    try:
        return Image.Resampling.BILINEAR
    except AttributeError:
        return Image.BILINEAR


def _decode_fourcc(value):
    try:
        value = int(value or 0)
        return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
    except Exception:
        return ""


def _detect(image):
    """Compatibility wrapper for ObjectDetection API variants."""
    try:
        return object_detection.detect(image, confidence=S.CONFIDENCE)
    except TypeError:
        try:
            return object_detection.detect(image, threshold=S.CONFIDENCE)
        except TypeError:
            return object_detection.detect(image)


def _configured_detection_interval():
    try:
        configured = float(
            getattr(S, "CONTINUOUS_DETECTION_INTERVAL", DEFAULT_DETECTION_INTERVAL)
        )
    except Exception:
        configured = DEFAULT_DETECTION_INTERVAL

    if configured <= 0:
        configured = DEFAULT_DETECTION_INTERVAL

    # Do not silently let an old 20+ Hz setting saturate inference resources
    return max(MIN_DETECTION_INTERVAL, configured)


# ============================================================================
# Camera discovery
# ============================================================================
_CAMERA_INDEX = None
_last_camera_log = None


def _clog(msg):
    """Log a camera message only when the message changes."""
    global _last_camera_log
    if msg != _last_camera_log:
        print(f"[camera] {msg}", flush=True)
        _last_camera_log = msg


def _find_usb_camera():
    if cv2 is None:
        return None

    candidates = []
    try:
        for name in os.listdir("/dev"):
            if name.startswith("video") and name[5:].isdigit():
                candidates.append(int(name[5:]))
    except Exception as exc:
        _clog(f"/dev listdir failed: {exc}")

    candidates.sort()
    if not candidates:
        _clog("no /dev/video* nodes present")
        return None

    for idx in candidates:
        cap = None
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                continue

            # Discovery remains conservative. We only accept a node after a
            # real decoded frame is returned.
            for _ in range(5):
                ok, frame = cap.read()
                if ok and frame is not None:
                    _clog(f"working camera at /dev/video{idx}")
                    return idx
                time.sleep(0.05)
        except Exception:
            pass
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

    _clog(
        f"no capture-capable camera among {candidates} "
        f"(these are likely ISP/codec nodes, not a USB webcam)"
    )
    return None


def reset_camera_index():
    global _CAMERA_INDEX
    _CAMERA_INDEX = None
    S.CAMERA_STATE["ok"] = False
    S.CAMERA_STATE["running"] = False
    S.CAMERA_STATE["error"] = "reset by user"


# ============================================================================
# Camera stream loop - high FPS, RAM only
# ============================================================================
def camera_stream_loop():
    global _CAMERA_INDEX

    if cv2 is None:
        S.CAMERA_STATE["ok"] = False
        S.CAMERA_STATE["running"] = False
        S.CAMERA_STATE["error"] = "OpenCV unavailable"
        return

    import state

    cap = None
    last_fps_time = time.monotonic()
    last_good = 0.0
    last_ui_encode = 0.0
    fps_counter = 0

    # Detection-result cache only. Crucially, the live WebUI image remains the
    # current camera frame. We do NOT cache an old annotated background because
    # that would make a 30 FPS capture stream look like a 5 FPS video stream.
    cached_overlay_results = None
    cached_overlay_ts = 0.0

    while True:
        try:
            S.heartbeat("camera")

            # ----------------------------------------------------------------
            # Find camera
            # ----------------------------------------------------------------
            if _CAMERA_INDEX is None:
                found = _find_usb_camera()
                if found is None:
                    S.CAMERA_STATE.update(
                        ok=False,
                        running=False,
                        index=None,
                        error=f"no camera; nodes={list_video_devices()}",
                    )
                    time.sleep(15.0)
                    continue

                _CAMERA_INDEX = found
                S.CAMERA_STATE["index"] = found

            # ----------------------------------------------------------------
            # Open and configure camera
            # ----------------------------------------------------------------
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(_CAMERA_INDEX, cv2.CAP_V4L2)

                if not cap.isOpened():
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                    _CAMERA_INDEX = None
                    S.CAMERA_STATE.update(
                        ok=False,
                        running=False,
                        index=None,
                        error="camera open failed",
                    )
                    time.sleep(2.0)
                    continue

                try:
                    # Pixel format first, then geometry and frame rate
                    cap.set(
                        cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc(*CAMERA_REQUEST_FOURCC),
                    )
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_REQUEST_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_REQUEST_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, CAMERA_REQUEST_FPS)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
                except Exception as config_exc:
                    print(
                        f"[camera] configuration warning: "
                        f"{type(config_exc).__name__}: {config_exc}",
                        flush=True,
                    )

                # Drain frames queued while changing V4L2 properties
                try:
                    for _ in range(CAMERA_OPEN_WARMUP_GRABS):
                        cap.grab()
                except Exception:
                    pass

                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                driver_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                actual_fourcc = _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
                reopens = int(S.CAMERA_STATE.get("reopens", 0) or 0) + 1

                S.CAMERA_STATE.update(
                    running=True,
                    ok=False,
                    index=_CAMERA_INDEX,
                    error=None,
                    reopens=reopens,
                    requested_fps=CAMERA_REQUEST_FPS,
                    requested_width=CAMERA_REQUEST_WIDTH,
                    requested_height=CAMERA_REQUEST_HEIGHT,
                    driver_fps=driver_fps,
                    actual_fps=driver_fps,
                    actual_width=actual_width,
                    actual_height=actual_height,
                    fourcc=actual_fourcc,
                    ui_target_fps=UI_TARGET_FPS,
                    detection_width=DETECTION_WIDTH,
                    detection_height=DETECTION_HEIGHT,
                    detection_interval_s=_configured_detection_interval(),
                )

                if reopens == 1:
                    S.robot_say(
                        f"USB camera stream started on /dev/video{_CAMERA_INDEX}."
                    )

                print(
                    f"[camera] opened /dev/video{_CAMERA_INDEX} "
                    f"requested={CAMERA_REQUEST_WIDTH}x{CAMERA_REQUEST_HEIGHT}"
                    f"@{CAMERA_REQUEST_FPS} {CAMERA_REQUEST_FOURCC} "
                    f"actual={actual_width}x{actual_height}@{driver_fps:.1f} "
                    f"fourcc={actual_fourcc or 'unknown'} "
                    f"detect={DETECTION_WIDTH}x{DETECTION_HEIGHT} "
                    f"ui_target={UI_TARGET_FPS:.1f}fps",
                    flush=True,
                )

                last_good = time.monotonic()
                last_fps_time = last_good
                last_ui_encode = 0.0
                fps_counter = 0
                cached_overlay_results = None
                cached_overlay_ts = 0.0

            # ----------------------------------------------------------------
            # Capture
            # ----------------------------------------------------------------
            ok, frame = cap.read()
            capture_now = time.monotonic()

            if not ok or frame is None:
                stall_age = capture_now - last_good
                if stall_age > float(S.CAMERA_STALL_TIMEOUT_SEC):
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                    _CAMERA_INDEX = None
                    S.CAMERA_STATE.update(
                        ok=False,
                        running=False,
                        error="stall - reopening",
                    )
                    continue

                S.CAMERA_STATE.update(ok=False, error="read failed")
                time.sleep(0.10)
                continue

            last_good = capture_now

            if S.CAMERA_ROTATE_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_clean = Image.fromarray(rgb)

            # ----------------------------------------------------------------
            # Publish smaller inference frame
            # ----------------------------------------------------------------
            if S.CONTINUOUS_DETECTION_ENABLED:
                detect_frame = pil_clean.resize(
                    (DETECTION_WIDTH, DETECTION_HEIGHT),
                    _bilinear_resampling(),
                )
                with S.live_frame_lock:
                    state.LIVE_FRAME_RGB = detect_frame

            # ----------------------------------------------------------------
            # Fetch most recent detection result
            # ----------------------------------------------------------------
            overlay_is_fresh = False
            if S.CONTINUOUS_DETECTION_ENABLED:
                with S.live_detections_lock:
                    raw = S.LIVE_DETECTIONS.get("_raw_results")
                    detection_ts = float(
                        S.LIVE_DETECTIONS.get("timestamp", 0.0) or 0.0
                    )

                if raw is not None and detection_ts > 0.0:
                    detection_age = time.time() - detection_ts
                    overlay_is_fresh = detection_age < float(
                        S.LIVE_OVERLAY_MAX_AGE_SEC
                    )
                    if overlay_is_fresh:
                        cached_overlay_results = raw
                        cached_overlay_ts = detection_ts

            # ----------------------------------------------------------------
            # WebUI encode at independent cadence
            # ----------------------------------------------------------------
            now_wall = time.time()
            if capture_now - last_ui_encode >= UI_FRAME_INTERVAL:
                display_pil = pil_clean

                # The helper is invoked only at the 10 FPS UI cadence, not on
                # every captured frame. Boxes are drawn against a 320x240 copy
                # and then enlarged, ensuring coordinates remain aligned with
                # the 320x240 inference image.
                if (
                    S.CONTINUOUS_DETECTION_ENABLED
                    and cached_overlay_results is not None
                    and cached_overlay_ts > 0.0
                    and (now_wall - cached_overlay_ts)
                    < float(S.LIVE_OVERLAY_MAX_AGE_SEC)
                ):
                    try:
                        overlay_source = pil_clean.resize(
                            (DETECTION_WIDTH, DETECTION_HEIGHT),
                            _bilinear_resampling(),
                        )
                        annotated_small = object_detection.draw_bounding_boxes(
                            overlay_source,
                            cached_overlay_results,
                        )
                        if annotated_small is not None:
                            display_pil = annotated_small.resize(
                                pil_clean.size,
                                _bilinear_resampling(),
                            )
                    except Exception as overlay_exc:
                        if not S.LIVE_DETECTIONS.get("_overlay_err_logged"):
                            print(
                                f"[overlay] {type(overlay_exc).__name__}: "
                                f"{overlay_exc}",
                                flush=True,
                            )
                            S.LIVE_DETECTIONS["_overlay_err_logged"] = True

                S.CAMERA_STATE["frame_base64"] = pil_to_jpeg_base64(
                    display_pil,
                    quality=UI_JPEG_QUALITY,
                )
                S.CAMERA_STATE["timestamp"] = now_wall
                last_ui_encode = capture_now

            # ----------------------------------------------------------------
            # Health and FPS
            # ----------------------------------------------------------------
            S.CAMERA_STATE["ok"] = True
            S.CAMERA_STATE["running"] = True
            S.CAMERA_STATE["error"] = None

            fps_counter += 1
            fps_window = capture_now - last_fps_time
            if fps_window >= 1.0:
                measured_capture_fps = fps_counter / fps_window
                S.CAMERA_STATE["fps"] = round(measured_capture_fps, 1)
                S.CAMERA_STATE["capture_fps"] = round(measured_capture_fps, 1)
                fps_counter = 0
                last_fps_time = capture_now

            # Intentionally no time.sleep(S.CAMERA_FRAME_INTERVAL). cap.read()
            # provides capture pacing; WebUI and AI have separate throttles.

        except Exception as exc:
            S.CAMERA_STATE.update(
                ok=False,
                running=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            print(
                f"[CAMERA ERROR] {type(exc).__name__}: {exc}",
                flush=True,
            )
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

            cap = None
            _CAMERA_INDEX = None
            cached_overlay_results = None
            cached_overlay_ts = 0.0
            time.sleep(2.0)


# ============================================================================
# Continuous detection loop
# ============================================================================
def continuous_detection_loop(autonomy_step_fn):
    import state

    last_run = 0.0
    previous_found = False

    while True:
        try:
            S.heartbeat("live-detect")

            if not S.CONTINUOUS_DETECTION_ENABLED:
                time.sleep(0.25)
                continue

            now = time.monotonic()
            interval = _configured_detection_interval()
            remaining = interval - (now - last_run)
            if remaining > 0:
                time.sleep(min(remaining, 0.025))
                continue

            # Copy the PIL object reference under lock. Camera publishing swaps
            # the entire object, so inference can safely work on the previous
            # immutable frame after the lock is released.
            with S.live_frame_lock:
                frame = state.LIVE_FRAME_RGB

            if frame is None:
                time.sleep(0.05)
                continue

            inference_frame = frame.copy()
            inference_started = time.monotonic()
            results = _detect(inference_frame)
            elapsed_ms = (time.monotonic() - inference_started) * 1000.0

            dets = extract_detections(results)
            target_norm = normalize_label(
                S.LIVE_DETECTIONS.get("target")
                or S.CONTINUOUS_DETECTION_TARGET
            )
            target_hits = [
                det
                for det in dets
                if normalize_label(detection_label(det)) == target_norm
            ]

            found_now = len(target_hits) > 0
            confidence = max(
                (detection_score(det) for det in target_hits),
                default=0.0,
            )
            results_ts = time.time()

            with S.live_detections_lock:
                S.LIVE_DETECTIONS["detections"] = dets
                S.LIVE_DETECTIONS["_raw_results"] = results
                S.LIVE_DETECTIONS["timestamp"] = results_ts
                S.LIVE_DETECTIONS["processing_time_ms"] = round(elapsed_ms, 1)
                S.LIVE_DETECTIONS["inference_fps"] = round(
                    1000.0 / elapsed_ms,
                    2,
                ) if elapsed_ms > 0 else 0.0
                S.LIVE_DETECTIONS["count"] = len(dets)
                S.LIVE_DETECTIONS["target_count"] = len(target_hits)
                S.LIVE_DETECTIONS["found_target"] = found_now
                S.LIVE_DETECTIONS["confidence"] = confidence
                S.LIVE_DETECTIONS["input_width"] = inference_frame.width
                S.LIVE_DETECTIONS["input_height"] = inference_frame.height
                S.LIVE_DETECTIONS["error"] = None

            if found_now and not previous_found:
                text = f"{target_norm.upper()} {int(confidence * 100)}%"
                HW.matrix_enqueue("flash", text)
                HW.oled_detection(text)
                HW.set_mood("detected")
                if hasattr(HW, "set_person_detected"):
                    HW.set_person_detected(True)
                else:
                    HW.bridge_call_quiet("set_person_detected", 1)
                if F is not None and hasattr(F, "emit"):
                    F.emit("person_detected", request_mcu=False)

            elif not found_now and previous_found:
                HW.matrix_enqueue("idle")
                HW.set_mood("searching")
                if hasattr(HW, "set_person_detected"):
                    HW.set_person_detected(False)
                else:
                    HW.bridge_call_quiet("set_person_detected", 0)

            S.LIVE_DETECTIONS["_prev_found_target"] = found_now

            try:
                img_w = inference_frame.width
                img_h = inference_frame.height
                target_det = target_hits[0] if target_hits else None
                try:
                    autonomy_step_fn(target_det, img_w, img_h)
                except TypeError:
                    autonomy_step_fn(target_det, img_w)
            except Exception as exc:
                print(
                    f"[live-detect] autonomy callback: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

            previous_found = found_now
            last_run = time.monotonic()

        except Exception as exc:
            S.LIVE_DETECTIONS["error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"[live-detect] error: {type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(1.0)


# ============================================================================
# One-shot detect
# ============================================================================
def get_latest_frame_or_capture():
    # Prefer the already-captured in-memory WebUI frame. This avoids opening a
    # second V4L2 handle while camera_stream_loop owns the USB camera.
    encoded = S.CAMERA_STATE.get("frame_base64")
    if encoded:
        raw = base64.b64decode(encoded)
        with Image.open(io.BytesIO(raw)) as image:
            return image.convert("RGB")

    if cv2 is None:
        raise RuntimeError("OpenCV not available")

    idx = _CAMERA_INDEX if _CAMERA_INDEX is not None else _find_usb_camera()
    if idx is None:
        raise RuntimeError(f"no camera; nodes={list_video_devices()}")

    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {idx}")

    ok = False
    frame = None
    try:
        try:
            cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*CAMERA_REQUEST_FOURCC),
            )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_REQUEST_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_REQUEST_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAMERA_REQUEST_FPS)
        except Exception:
            pass

        for _ in range(5):
            cap.grab()
        ok, frame = cap.read()
    finally:
        cap.release()

    if not ok or frame is None:
        raise RuntimeError("capture failed")

    if S.CAMERA_ROTATE_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def run_object_detection(target, language, autonomy_apply_fn):
    target = normalize_label(target)
    language = unquote(language).strip()

    S.STATE["target_object"] = target
    S.STATE["language"] = language

    HW.matrix_enqueue("scroll", "LOOKING", 35)
    HW.set_mood("searching")
    S.robot_say(f"Looking for {target}. Please wait.")

    try:
        pil_image = get_latest_frame_or_capture()

        # One-shot detection retains display resolution for compatibility with
        # the existing annotated API response. Continuous detection is the
        # performance-sensitive path and uses 320x240 above.
        start = time.monotonic()
        results = _detect(pil_image)
        elapsed_ms = (time.monotonic() - start) * 1000.0

        detections = extract_detections(results)
        target_hits = [
            det
            for det in detections
            if normalize_label(detection_label(det)) == target
        ]
        found = len(target_hits) > 0

        try:
            annotated = object_detection.draw_bounding_boxes(pil_image, results)
        except Exception:
            annotated = None

        if annotated is None:
            annotated = pil_image

        image_b64 = pil_to_jpeg_base64(annotated, quality=70)
        simplified = [
            {
                "label": detection_label(det),
                "score": detection_score(det),
                "raw": det,
            }
            for det in detections
        ]

        S.STATE["last_detection"] = {
            "ok": True,
            "target": target,
            "found": found,
            "count": len(detections),
            "target_count": len(target_hits),
            "processing_time_ms": round(elapsed_ms, 2),
            "image_base64": image_b64,
            "detections": simplified,
        }

        if found:
            score = int(detection_score(target_hits[0]) * 100)
            msg = f"{target.upper()} {score}%"
            HW.matrix_enqueue("flash", msg)
            HW.oled_detection(msg)
            HW.set_mood("detected")

            if hasattr(HW, "set_person_detected"):
                HW.set_person_detected(True)
            else:
                HW.bridge_call_quiet("set_person_detected", 1)

            if F is not None and hasattr(F, "emit"):
                F.emit("person_detected", request_mcu=False)

            img_w, _ = pil_image.size
            cx = detection_center_x(target_hits[0], img_w)
            autonomy_apply_fn(cx, img_w)
            S.robot_say(
                f"I found {target}. Detections: {len(detections)}."
            )
        else:
            HW.matrix_enqueue("scroll", "NO TARGET", 35)
            HW.oled_detection("SEARCHING")
            S.robot_say(
                f"I did not find {target} (saw {len(detections)} others)."
            )

        return S.STATE["last_detection"]

    except Exception as exc:
        S.STATE["last_detection"] = {
            "ok": False,
            "target": target,
            "found": False,
            "error": f"{type(exc).__name__}: {exc}",
            "image_base64": None,
            "detections": [],
        }
        S.robot_say(f"Object detection failed: {exc}")
        return S.STATE["last_detection"]
