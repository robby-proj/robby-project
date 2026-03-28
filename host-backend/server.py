def _extract_answer(self, raw: str) -> str:
    text = raw

    # Remove the final USER> prompt, if present
    end_idx = text.rfind(PROMPT_MARKER)
    if end_idx != -1:
        text = text[:end_idx]

    # If the answer starts immediately after USER>, keep the text after it
    first_idx = text.find(PROMPT_MARKER)
    if first_idx != -1:
        text = text[first_idx + len(PROMPT_MARKER):]

    lines = text.splitlines()
import json
import os
import queue
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, unquote

HOST = "0.0.0.0"
PORT = 8090

YZMA_DIR = "/home/arduino/yzma"
MODEL_PATH = "/home/arduino/models/SmolLM2-135M-Instruct-Q4_K_M.gguf"
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
STARTUP_TIMEOUT = 240
RESPONSE_TIMEOUT = 180


class YzmaSession:
    def __init__(self):
        self.proc = None
        self.q = queue.Queue()
        self.lock = threading.Lock()

    def _reader(self):
        try:
            while True:
                if self.proc is None or self.proc.stdout is None:
                    return

                chunk = self.proc.stdout.read(1)
                if not chunk:
                    return

                self.q.put(chunk)
                print(chunk, end="", flush=True)
        except Exception as e:
            print(f"[reader] {e}", flush=True)

    def _ensure_paths(self):
        if not os.path.isdir(YZMA_DIR):
            raise RuntimeError(f"YZMA_DIR not found: {YZMA_DIR}")
        if not os.path.isfile(MODEL_PATH):
            raise RuntimeError(f"MODEL_PATH not found: {MODEL_PATH}")
        if not os.path.isdir(LIB_DIR):
            raise RuntimeError(f"LIB_DIR not found: {LIB_DIR}")

    def _drain_queue(self):
        drained = []
        while True:
            try:
                drained.append(self.q.get_nowait())
            except queue.Empty:
                break
        return "".join(drained)

    def _read_until_n_prompts(self, prompt_count_needed, timeout):
        deadline = time.time() + timeout
        chunks = []
        joined = ""

        while time.time() < deadline:
            try:
                part = self.q.get(timeout=0.25)
                chunks.append(part)
                joined += part

                if joined.count(PROMPT_MARKER) >= prompt_count_needed:
                    return joined
            except queue.Empty:
                if self.proc is not None and self.proc.poll() is not None:
                    break

        return "".join(chunks)

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
            bufsize=0,
            env=env,
        )

        threading.Thread(target=self._reader, daemon=True).start()

        startup_text = self._read_until_n_prompts(
            prompt_count_needed=1,
            timeout=STARTUP_TIMEOUT,
        )

        if PROMPT_MARKER not in startup_text:
            raise RuntimeError(
                "yzma started, but USER> prompt was not detected. "
                f"Startup output tail:\n{startup_text[-1200:]}"
            )

    def ask(self, prompt: str) -> str:
        with self.lock:
            self.start()

            if self.proc is None or self.proc.poll() is not None:
                raise RuntimeError("yzma process exited unexpectedly")

            self._drain_queue()

            assert self.proc.stdin is not None
            self.proc.stdin.write(prompt + "\n")
            self.proc.stdin.flush()

            raw = self._read_until_n_prompts(
                prompt_count_needed=1,
                timeout=RESPONSE_TIMEOUT,
            )

            return self._extract_answer(raw)

    def _extract_answer(self, raw: str) -> str:
        text = raw

        end_idx = text.rfind(PROMPT_MARKER)
        if end_idx != -1:
            text = text[:end_idx]

        first_idx = text.find(PROMPT_MARKER)
        if first_idx != -1:
            text = text[first_idx + len(PROMPT_MARKER):]

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

            if line.startswith(PROMPT_MARKER):
                line = line[len(PROMPT_MARKER):].lstrip()
                if not line:
                    continue

            if "HTTP/1.1" in line and "/api/chat/" in line:
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
        self._drain_queue()


yzma = YzmaSession()


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            print("Client disconnected before response was sent", flush=True)

    def do_OPTIONS(self):
        self._send({}, 200)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/health":
            errors = []
            if not os.path.isdir(YZMA_DIR):
                errors.append(f"Missing yzma dir: {YZMA_DIR}")
            if not os.path.isfile(MODEL_PATH):
                errors.append(f"Missing model: {MODEL_PATH}")
            if not os.path.isdir(LIB_DIR):
                errors.append(f"Missing lib dir: {LIB_DIR}")

            self._send({
                "ok": len(errors) == 0,
                "errors": errors,
                "yzma_dir": YZMA_DIR,
                "model_path": MODEL_PATH,
                "lib_dir": LIB_DIR,
            })
            return

        if parsed.path == "/api/restart":
            try:
                yzma.stop()
                self._send({"ok": True})
            except Exception as e:
                self._send({"ok": False, "error": str(e)}, 500)
            return

        if parsed.path.startswith("/api/chat/"):
            try:
                prompt = unquote(parsed.path[len("/api/chat/"):]).strip()
                if not prompt:
                    self._send({"ok": False, "error": "Prompt is empty"}, 400)
                    return

                answer = yzma.ask(prompt)
                self._send({"ok": True, "answer": answer})
            except Exception as e:
                self._send({"ok": False, "error": str(e)}, 500)
            return

        self._send({"ok": False, "error": "Not found"}, 404)


if __name__ == "__main__":
    print(f"Host backend listening on http://{HOST}:{PORT}", flush=True)
    HTTPServer((HOST, PORT), Handler).serve_forever()
