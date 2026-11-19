from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_DIR))

from diagnostics import build_diagnostic_context  # noqa: E402


class DiagnosticContextTests(unittest.TestCase):
    def test_structured_parameters_include_units_and_status(self):
        context = build_diagnostic_context(
            {
                "status": "warning",
                "parameters": {
                    "temperature": {
                        "label": "Temperature",
                        "value": 72.4,
                        "unit": "C",
                        "status": "warning",
                    }
                },
            },
            application_name="Test app",
        )
        self.assertIn("Temperature=72.4 C; status=warning", context)

    def test_arbitrary_json_is_flattened_and_secrets_are_removed(self):
        context = build_diagnostic_context(
            {
                "machine": {
                    "speed": 1200,
                    "api_key": "must-not-leak",
                }
            },
            application_name="Test app",
        )
        self.assertIn("machine.speed: 1200", context)
        self.assertNotIn("must-not-leak", context)
        self.assertNotIn("api_key", context)

    def test_stale_timestamp_is_marked(self):
        context = build_diagnostic_context(
            {"timestamp": "2026-11-18T12:00:00Z", "value": 1},
            application_name="Test app",
            stale_after_seconds=60,
            now=datetime(2026, 11, 18, 12, 2, tzinfo=timezone.utc),
        )
        self.assertIn("freshness=stale", context)

    def test_context_is_bounded(self):
        context = build_diagnostic_context(
            {"data": "x" * 2000},
            application_name="Test app",
            max_chars=200,
        )
        self.assertLessEqual(len(context), 200)
        self.assertTrue(context.endswith("[context truncated]"))


if __name__ == "__main__":
    unittest.main()
