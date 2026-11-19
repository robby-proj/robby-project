"""Diagnostic snapshot adapters for applications connected to VENTUNO Q."""

from __future__ import annotations

import json
import threading
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping


class SnapshotSource(ABC):
    @abstractmethod
    def read(self) -> Mapping[str, Any]:
        """Return the latest application snapshot."""


class JsonFileSource(SnapshotSource):
    def __init__(self, path: Path):
        self.path = path

    def read(self) -> Mapping[str, Any]:
        with self.path.open("r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
        if not isinstance(snapshot, dict):
            raise ValueError("The JSON file must contain an object at its root")
        return snapshot


class HttpJsonSource(SnapshotSource):
    def __init__(self, url: str, timeout_seconds: float = 3.0):
        self.url = url
        self.timeout_seconds = timeout_seconds

    def read(self) -> Mapping[str, Any]:
        request = urllib.request.Request(
            self.url,
            headers={"Accept": "application/json", "User-Agent": "ventuno-q-diagnostics/1.0"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            snapshot = json.loads(response.read().decode("utf-8"))
        if not isinstance(snapshot, dict):
            raise ValueError("The HTTP endpoint must return a JSON object")
        return snapshot


class MqttJsonSource(SnapshotSource):
    """Maintain the most recent JSON snapshot published on an MQTT topic."""

    def __init__(
        self,
        host: str,
        port: int,
        topic: str,
        *,
        username: str | None = None,
        password: str | None = None,
    ):
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError("MQTT source requires paho-mqtt") from exc

        self._lock = threading.Lock()
        self._latest: Mapping[str, Any] | None = None
        self._client = mqtt.Client()
        if username:
            self._client.username_pw_set(username, password)

        def on_connect(client, userdata, flags, rc):
            if rc != 0:
                print(f"MQTT connection failed with code {rc}")
                return
            client.subscribe(topic)
            print(f"MQTT source subscribed to {topic}")

        def on_message(client, userdata, message):
            try:
                snapshot = json.loads(message.payload.decode("utf-8"))
                if not isinstance(snapshot, dict):
                    raise ValueError("root value is not an object")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                print(f"Ignored invalid MQTT diagnostic snapshot: {exc}")
                return
            with self._lock:
                self._latest = snapshot

        self._client.on_connect = on_connect
        self._client.on_message = on_message
        self._client.connect_async(host, port)
        self._client.loop_start()

    def read(self) -> Mapping[str, Any]:
        with self._lock:
            if self._latest is None:
                raise RuntimeError("No diagnostic snapshot has arrived on the MQTT topic yet")
            return dict(self._latest)


def create_source(config: Mapping[str, Any], app_root: Path) -> SnapshotSource:
    source_type = str(config.get("type", "file")).lower()
    if source_type == "file":
        path = Path(str(config.get("path", "examples/status.json")))
        if not path.is_absolute():
            path = app_root / path
        return JsonFileSource(path)
    if source_type == "http":
        return HttpJsonSource(
            str(config["url"]),
            float(config.get("timeout_seconds", 3.0)),
        )
    if source_type == "mqtt":
        return MqttJsonSource(
            str(config.get("host", "127.0.0.1")),
            int(config.get("port", 1883)),
            str(config["topic"]),
            username=config.get("username"),
            password=config.get("password"),
        )
    raise ValueError(f"Unsupported source type: {source_type}")
