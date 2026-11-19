"""Minimal HTTP data source used to test the diagnostic chat integration."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class StatusHandler(BaseHTTPRequestHandler):
    status_file: Path

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/status":
            self.send_error(404)
            return
        try:
            payload = json.loads(self.status_file.read_text(encoding="utf-8"))
            body = json.dumps(payload).encode("utf-8")
        except (OSError, json.JSONDecodeError) as exc:
            self.send_error(500, str(exc))
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a diagnostic JSON snapshot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(__file__).with_name("status.json"),
    )
    args = parser.parse_args()
    StatusHandler.status_file = args.file.resolve()
    server = ThreadingHTTPServer((args.host, args.port), StatusHandler)
    print(f"Serving {StatusHandler.status_file} at http://{args.host}:{args.port}/status")
    server.serve_forever()


if __name__ == "__main__":
    main()
