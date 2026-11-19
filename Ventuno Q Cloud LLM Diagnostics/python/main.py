"""VENTUNO Q local diagnostic LLM exposed through Arduino Cloud Messenger."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from arduino.app_bricks.arduino_cloud import ArduinoCloud, CLOUD_WINS
from arduino.app_bricks.llm import LargeLanguageModel
from arduino.app_utils import App

from diagnostics import DEFAULT_DENY_PATTERNS, build_diagnostic_context
from sources import create_source


APP_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(
    os.environ.get("DIAGNOSTICS_CONFIG", str(APP_ROOT / "config" / "diagnostics.json"))
)

with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    CONFIG = json.load(handle)

APPLICATION_NAME = str(CONFIG.get("application_name", "Connected application"))
CHAT_CONFIG = CONFIG.get("chat", {})
PRIVACY_CONFIG = CONFIG.get("privacy", {})
CLOUD_VARIABLE = str(CHAT_CONFIG.get("cloud_variable", "chat_message"))
REPLY_MAX_CHARS = int(CHAT_CONFIG.get("reply_max_chars", 220))
QUESTION_MAX_CHARS = int(CHAT_CONFIG.get("question_max_chars", 500))
CONTEXT_MAX_CHARS = int(CHAT_CONFIG.get("context_max_chars", 6000))
STALE_AFTER_SECONDS = int(CHAT_CONFIG.get("stale_after_seconds", 120))
ALLOW_PATHS = tuple(PRIVACY_CONFIG.get("allow_paths", ()))
DENY_PATTERNS = tuple(PRIVACY_CONFIG.get("deny_key_patterns", DEFAULT_DENY_PATTERNS))

SYSTEM_PROMPT = (
    "You are an on-device diagnostic assistant for an application connected to a VENTUNO Q. "
    "Answer only from the diagnostic snapshot supplied with the question. Do not invent values, "
    "causes, or actions. Distinguish observed facts from likely interpretations. If data is stale, "
    "missing, or insufficient, say so. Prefer one practical next check when appropriate. Reply in "
    "plain text only, with no markdown or line breaks, and stay under "
    f"{REPLY_MAX_CHARS} characters."
)

cloud = ArduinoCloud()
llm = LargeLanguageModel(system_prompt=SYSTEM_PROMPT)
source = create_source(CONFIG.get("source", {}), APP_ROOT)

_inference_lock = threading.Lock()
_state_lock = threading.Lock()
_last_device_reply = ""


def _publish_reply(reply: str) -> None:
    global _last_device_reply
    clean_reply = " ".join(str(reply).split())[:REPLY_MAX_CHARS]
    if not clean_reply:
        clean_reply = "No diagnostic answer is available."
    with _state_lock:
        _last_device_reply = clean_reply
    setattr(cloud, CLOUD_VARIABLE, clean_reply)
    print(f"Published diagnostic reply ({len(clean_reply)} characters)")


def _answer(question: str) -> None:
    if not _inference_lock.acquire(blocking=False):
        _publish_reply("A diagnostic request is already running. Please try again in a moment.")
        return
    try:
        try:
            snapshot = source.read()
            context = build_diagnostic_context(
                snapshot,
                application_name=APPLICATION_NAME,
                max_chars=CONTEXT_MAX_CHARS,
                stale_after_seconds=STALE_AFTER_SECONDS,
                allow_paths=ALLOW_PATHS,
                deny_patterns=DENY_PATTERNS,
            )
            prompt = f"DIAGNOSTIC SNAPSHOT:\n{context}\n\nUSER QUESTION:\n{question}"
            reply = llm.chat(prompt)
        except Exception as exc:
            print(f"Diagnostic request failed: {type(exc).__name__}: {exc}")
            reply = "I could not read the latest diagnostic data. Check the local data source and try again."
        _publish_reply(reply)
    finally:
        _inference_lock.release()


def on_chat_message(client, value) -> None:
    """Handle a user prompt written by the Arduino Cloud Messenger widget."""
    if not isinstance(value, str):
        return
    question = value.strip()
    if not question:
        return

    with _state_lock:
        if question == _last_device_reply:
            return

    question = question[:QUESTION_MAX_CHARS]
    threading.Thread(target=_answer, args=(question,), daemon=True).start()


cloud.register(CLOUD_VARIABLE, value="", sync=CLOUD_WINS, on_write=on_chat_message)

print(f"Diagnostic chat is ready on Arduino Cloud variable '{CLOUD_VARIABLE}'")
App.run()
