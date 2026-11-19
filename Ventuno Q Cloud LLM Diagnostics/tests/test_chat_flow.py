from __future__ import annotations

import importlib.util
import sys
import threading
import types
import unittest
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
sys.path.insert(0, str(PYTHON_DIR))


class FakeCloud:
    def __init__(self):
        self.callbacks = {}
        self.reply_event = threading.Event()

    def register(self, name, value, sync, on_write):
        self.callbacks[name] = on_write
        setattr(self, name, value)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == "chat_message" and value:
            event = getattr(self, "reply_event", None)
            if event:
                event.set()


class FakeLlm:
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt
        self.last_prompt = ""

    def chat(self, prompt):
        self.last_prompt = prompt
        return "Motor temperature is above its preferred range."


class FakeApp:
    @staticmethod
    def run():
        return None


def load_main_with_brick_stubs():
    arduino = types.ModuleType("arduino")
    app_utils = types.ModuleType("arduino.app_utils")
    app_bricks = types.ModuleType("arduino.app_bricks")
    cloud_brick = types.ModuleType("arduino.app_bricks.arduino_cloud")
    llm_brick = types.ModuleType("arduino.app_bricks.llm")

    app_utils.App = FakeApp
    cloud_brick.ArduinoCloud = FakeCloud
    cloud_brick.CLOUD_WINS = "cloud_wins"
    llm_brick.LargeLanguageModel = FakeLlm

    replacements = {
        "arduino": arduino,
        "arduino.app_utils": app_utils,
        "arduino.app_bricks": app_bricks,
        "arduino.app_bricks.arduino_cloud": cloud_brick,
        "arduino.app_bricks.llm": llm_brick,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)

    module_name = f"diagnostic_chat_main_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, PYTHON_DIR / "main.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    return module


class ChatFlowTests(unittest.TestCase):
    def test_question_builds_context_and_publishes_short_reply(self):
        module = load_main_with_brick_stubs()
        callback = module.cloud.callbacks["chat_message"]
        callback(None, "What is abnormal right now?")

        self.assertTrue(module.cloud.reply_event.wait(timeout=2))
        self.assertEqual(
            module.cloud.chat_message,
            "Motor temperature is above its preferred range.",
        )
        self.assertIn("DIAGNOSTIC SNAPSHOT:", module.llm.last_prompt)
        self.assertIn("Motor temperature=72.4 C", module.llm.last_prompt)
        self.assertIn("USER QUESTION:\nWhat is abnormal right now?", module.llm.last_prompt)

    def test_device_reply_is_not_processed_as_a_new_question(self):
        module = load_main_with_brick_stubs()
        module._publish_reply("Current state is normal.")
        module.llm.last_prompt = ""

        module.on_chat_message(None, "Current state is normal.")

        self.assertEqual(module.llm.last_prompt, "")


if __name__ == "__main__":
    unittest.main()
