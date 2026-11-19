"""audio_feedback.py - Robodog Pro v8.4.6 USB speaker and MCU buzzer event worker.

Plays bundled, original synthesized WAV cues through ALSA `aplay`. The worker
is optional and fail-soft: when no USB playback device is present the rest of
Robodog continues normally. No audio is recorded and no runtime files are
created. Safety events preempt decorative sounds.
"""
from __future__ import annotations
import heapq, os, re, shutil, subprocess, threading, time
from pathlib import Path
import state as S

AUDIO_DIR = Path(__file__).resolve().parent / "audio"
APLAY = shutil.which("aplay")
_ALSA_DEVICE = None

def _alsa_devices():
    """Return explicit ALSA playback PCMs, preferring USB/C-Media devices."""
    override = os.environ.get("ROBODOG_ALSA_DEVICE", "").strip()
    if override:
        return [override]
    if not APLAY:
        return []
    found=[]
    try:
        r=subprocess.run([APLAY,"-l"],capture_output=True,text=True,timeout=3)
        for line in r.stdout.splitlines():
            m=re.search(r"card\s+(\d+):.*device\s+(\d+):",line,re.I)
            if not m: continue
            card,dev=int(m.group(1)),int(m.group(2))
            score=0 if any(x in line.lower() for x in ("usb","c-media","pnp")) else 1
            found.append((score,card,dev))
    except Exception:
        return []
    devices=[]
    for _,card,dev in sorted(set(found)):
        devices.extend([f"plughw:{card},{dev}",f"hw:{card},{dev}"])
    return devices

def _choose_alsa_device():
    devices=_alsa_devices()
    return devices[0] if devices else None

_lock = threading.Lock()
_wake = threading.Event()
_queue = []
_seq = 0
_current = None
_proc = None
_last = {}

BUZZER_EVENT_IDS = {
    "boot_complete": 1, "controller_connected": 2, "controller_armed": 3,
    "controller_disconnected": 4, "autonomous_started": 5,
    "walkaround_started": 6, "person_detected": 7, "look_for_food": 8,
    "creep_forward": 9, "sit_beg": 10, "dance": 11, "collision": 12,
    "hard_stop": 13, "low_battery": 14, "bridge_lost": 15,
    "bridge_recovered": 16, "sleep": 17,
    "manual_mode": 18, "autonomous_mode": 19, "wall_hit": 20,
    "wake": 21, "sleep_confirm": 22, "action_confirm": 23,
    "greeting": 24, "dance_extended": 25, "motors_on": 26,
    "motors_off": 27, "mode_confirm": 28, "error": 29,
}

EVENTS = {
    "boot_complete": (40, "startup.wav", 2.0),
    "controller_connected": (45, "curious.wav", 2.0),
    "controller_armed": (35, "startup.wav", 2.0),
    "controller_disconnected": (80, "bridge_lost.wav", 2.0),
    "autonomous_started": (45, "curious.wav", 2.0),
    "walkaround_started": (45, "bark_double.wav", 2.0),
    "person_detected": (35, "bark_short.wav", 8.0),
    "look_for_food": (30, "sniff.wav", 1.0),
    "creep_forward": (30, "pant.wav", 1.0),
    "sit_beg": (30, "curious.wav", 1.0),
    "dance": (30, "playful.wav", 1.0),
    "collision": (95, "alert.wav", 1.0),
    "hard_stop": (100, "hard_stop.wav", 0.5),
    "low_battery": (70, "low_battery.wav", 60.0),
    "bridge_lost": (90, "bridge_lost.wav", 2.0),
    "bridge_recovered": (55, "startup.wav", 2.0),
    "sleep": (20, "sleep.wav", 5.0),
    "manual_mode": (40, "startup.wav", 1.0),
    "autonomous_mode": (45, "curious.wav", 1.0),
    "wall_hit": (95, "alert.wav", 1.0),
    "wake": (35, "startup.wav", 2.0),
    "sleep_confirm": (20, "sleep.wav", 5.0),
    "action_confirm": (25, "curious.wav", 0.4),
    "greeting": (35, "bark_short.wav", 1.0),
    "dance_extended": (35, "playful.wav", 1.0),
    "motors_on": (45, "startup.wav", 1.0),
    "motors_off": (55, "sleep.wav", 1.0),
    "mode_confirm": (30, "curious.wav", 0.8),
    "error": (90, "alert.wav", 1.0),
}

def _state():
    return S.STATE.setdefault("audio", {"speaker_available": False, "muted": False,
        "current": None, "last_event": None, "queue_depth": 0, "error": None})

def emit(event, request_mcu=True):
    global _seq
    spec=EVENTS.get(str(event))
    if not spec: return False
    priority, filename, cooldown=spec; now=time.monotonic()
    if now-_last.get(event,0)<cooldown: return False
    _last[event]=now
    # Always request the optional Modulino Buzzer pattern. The MCU provider is
    # fail-soft when the buzzer is absent, while a USB speaker may play in parallel.
    try:
        import hw_bridge as HW
        event_id = BUZZER_EVENT_IDS.get(str(event))
        if request_mcu and event_id is not None:
            HW.bridge_call_quiet("sound_event", event_id)
    except Exception:
        pass
    with _lock:
        _seq+=1; heapq.heappush(_queue,(-priority,_seq,str(event),filename))
        _state()["queue_depth"]=len(_queue)
    if priority>=90: stop_current()
    _wake.set(); return True

def stop_current():
    global _proc
    p=_proc
    if p and p.poll() is None:
        try: p.terminate()
        except Exception: pass

def status(): return dict(_state())

def set_muted(value): _state()["muted"]=bool(value); stop_current(); return status()

def _play(path):
    global _proc, _ALSA_DEVICE
    st=_state()
    if st.get("muted") or not APLAY: return
    candidates=[]
    if _ALSA_DEVICE: candidates.append(_ALSA_DEVICE)
    candidates.extend(d for d in _alsa_devices() if d not in candidates)
    if not candidates:
        raise RuntimeError("no ALSA playback PCM found (aplay -l returned none)")
    errors=[]
    for device in candidates:
        st["device"]=device
        _proc=subprocess.Popen([APLAY,"-q","-D",device,str(path)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        _,err=_proc.communicate(timeout=15)
        if _proc.returncode==0:
            _ALSA_DEVICE=device; st["speaker_available"]=True; return
        errors.append(f"{device}: {err.decode('utf-8','replace').strip()}")
    st["speaker_available"]=False
    raise RuntimeError("; ".join(errors))

def worker_loop():
    global _ALSA_DEVICE
    st=_state(); _ALSA_DEVICE=_choose_alsa_device(); st["device"]=_ALSA_DEVICE
    st["speaker_available"]=bool(APLAY and _ALSA_DEVICE)
    while True:
        S.heartbeat("audio")
        item=None
        with _lock:
            if _queue: item=heapq.heappop(_queue); st["queue_depth"]=len(_queue)
        if not item:
            _wake.wait(.5);_wake.clear();continue
        _priority,_n,event,filename=item; path=AUDIO_DIR/filename
        st.update(current=event,last_event=event,error=None)
        try:
            if not path.is_file(): raise FileNotFoundError(path)
            _play(path)
        except Exception as exc:
            st["error"]=str(exc); st["speaker_available"]=False
            print(f"[audio] {event} failed: {exc}",flush=True)
        finally: st["current"]=None
