import os
import queue
import subprocess
import threading
import time
from urllib.parse import unquote

from arduino.app_utils import App
from arduino.app_bricks.web_ui import WebUI

ui = WebUI()

YZMA_DIR = os.path.expanduser("~/yzma")
MODEL_PATH = os.path.expanduser("~/models/SmolLM2-135M-Instruct-Q4_K_M.gguf")
LIB_DIR = os.path.join(YZMA_DIR, "lib")
CHAT_CMD = [
    "go",
    "run",
    "./examples/chat/",
    "-model",
    MODEL_PATH,
    "-lib",
    "./lib/",
    "-v",
]

PROMPT_MARKER = "USER>"
STARTUP_TIMEOUT = 120
RESPONSE_TIMEOUT = 120


class YzmaSession:
    def __init__(self):
        self.proc = None
        self.q = queue.Queue()
        self.lock = threading.Lock()

    def _reader(self):
        try:
            while True:
                if self.proc is None:
                    return
                line = self.proc.stdout.readline()
                if not line:
                    return
                self.q.put(line)
                print(line, end="")
        except Exception as e:
            print(f"[reader] {e}", flush=True)

    def _ensure_paths(self):
        if not os.path.isdir(YZMA_DIR):
            raise RuntimeError(f"YZMA_DIR not found: {YZMA_DIR}")
        if not os.path.isfile(MODEL_PATH):
            raise RuntimeError(f"MODEL_PATH not found: {MODEL_PATH}")
        if not os.path.isdir(LIB_DIR):
            raise RuntimeError(f"LIB_DIR not found: {LIB_DIR}")

    def start(self):
        self._ensure_paths()

        if self.proc is not None and self.proc.poll() is None:
            return

        env = os.environ.copy()
        env["YZMA_LIB"] = LIB_DIR

        self.proc = subprocess.Popen(
            CHAT_CMD,
            cwd=YZMA_DIR,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        threading.Thread(target=self._reader, daemon=True).start()

        startup_text = self._read_until_prompt(timeout=STARTUP_TIMEOUT)
        if PROMPT_MARKER not in startup_text:
            raise RuntimeError("yzma started, but USER> prompt was not detected")

    def _read_until_prompt(self, timeout=RESPONSE_TIMEOUT):
        deadline = time.time() + timeout
        chunks = []

        while time.time() < deadline:
            try:
                part = self.q.get(timeout=0.25)
                chunks.append(part)
                joined = "".join(chunks)
                if PROMPT_MARKER in joined:
                    return joined
            except queue.Empty:
                pass

        return "".join(chunks)

    def ask(self, prompt: str) -> str:
        with self.lock:
            self.start()

            if self.proc.poll() is not None:
                raise RuntimeError("yzma process exited unexpectedly")

            self.proc.stdin.write(prompt + "\n")
            self.proc.stdin.flush()

            raw = self._read_until_prompt(timeout=RESPONSE_TIMEOUT)
            return self._extract_answer(raw)

    def _extract_answer(self, raw: str) -> str:
        text = raw
        idx = text.find(PROMPT_MARKER)
        if idx != -1:
            text = text[:idx]

        lines = text.splitlines()
        filtered = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                filtered.append("")
                continue

            if stripped.startswith("llama_"):
                continue
            if stripped.startswith("load_tensors:"):
                continue
            if stripped.startswith("graph_"):
                continue
            if stripped.startswith("sched_"):
                continue
            if stripped.startswith("main:"):
                continue
            if stripped.startswith("system_info:"):
                continue
            if stripped.startswith("build:"):
                continue
            if stripped.startswith("cpu_"):
                continue
            if stripped.startswith("AVX"):
                continue
            if stripped.startswith("USER>"):
                continue

            filtered.append(line)

        answer = "\n".join(filtered).strip()
        return answer if answer else "(No response text parsed.)"

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
        self.proc = None
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except Exception:
                break


yzma = YzmaSession()


def api_health():
    errors = []

    if not os.path.isdir(YZMA_DIR):
        errors.append(f"Missing yzma dir: {YZMA_DIR}")
    if not os.path.isfile(MODEL_PATH):
        errors.append(f"Missing model: {MODEL_PATH}")
    if not os.path.isdir(LIB_DIR):
        errors.append(f"Missing lib dir: {LIB_DIR}")

    return {
        "ok": len(errors) == 0,
        "yzma_dir": YZMA_DIR,
        "model_path": MODEL_PATH,
        "lib_dir": LIB_DIR,
        "errors": errors,
    }


def api_chat(prompt: str):
    try:
        prompt = unquote(prompt).strip()
        print(f"api_chat prompt={prompt!r}", flush=True)

        if not prompt:
            return {"ok": False, "error": "Prompt is empty"}

        answer = yzma.ask(prompt)
        return {"ok": True, "answer": answer}
    except Exception as e:
        print(f"api_chat error: {e}", flush=True)
        return {"ok": False, "error": str(e)}


def api_restart():
    try:
        yzma.stop()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


ui.expose_api("GET", "/api/health", api_health)
ui.expose_api("GET", "/api/chat/{prompt}", api_chat)
ui.expose_api("GET", "/api/restart", api_restart)


def loop():
    time.sleep(5)


App.run(user_loop=loop)