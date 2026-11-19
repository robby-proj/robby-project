"""Normalize arbitrary application data into compact, safe LLM context."""

from __future__ import annotations

import fnmatch
import json
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


DEFAULT_DENY_PATTERNS = (
    "password",
    "passwd",
    "token",
    "secret",
    "api_key",
    "apikey",
    "credential",
    "private_key",
)


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _matches(path: str, patterns: Iterable[str]) -> bool:
    lowered_path = path.lower()
    leaf = lowered_path.rsplit(".", 1)[-1]
    return any(
        fnmatch.fnmatch(lowered_path, pattern.lower())
        or fnmatch.fnmatch(leaf, pattern.lower())
        for pattern in patterns
    )


def _permitted(path: str, allow_paths: Iterable[str]) -> bool:
    patterns = tuple(allow_paths)
    if not patterns:
        return True
    lowered_path = path.lower()
    return any(fnmatch.fnmatch(lowered_path, pattern.lower()) for pattern in patterns)


def _compact(value: Any, max_chars: int = 300) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    else:
        text = str(value)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def _flatten(
    value: Any,
    prefix: str = "",
    *,
    allow_paths: Iterable[str] = (),
    deny_patterns: Iterable[str] = DEFAULT_DENY_PATTERNS,
) -> list[tuple[str, Any]]:
    """Flatten nested JSON while excluding likely credentials and secrets."""
    items: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if _matches(path, deny_patterns):
                continue
            items.extend(
                _flatten(
                    child,
                    path,
                    allow_paths=allow_paths,
                    deny_patterns=deny_patterns,
                )
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]"
            items.extend(
                _flatten(
                    child,
                    path,
                    allow_paths=allow_paths,
                    deny_patterns=deny_patterns,
                )
            )
    elif prefix and _permitted(prefix, allow_paths):
        items.append((prefix, value))
    return items


def _structured_parameter_lines(
    parameters: Mapping[str, Any],
    *,
    allow_paths: Iterable[str],
    deny_patterns: Iterable[str],
) -> list[str]:
    lines: list[str] = []
    for key, raw in parameters.items():
        path = f"parameters.{key}"
        if _matches(path, deny_patterns) or not _permitted(path, allow_paths):
            continue
        if not isinstance(raw, Mapping):
            lines.append(f"- {key}: {_compact(raw)}")
            continue

        label = raw.get("label", key)
        value = _compact(raw.get("value"))
        unit = f" {raw['unit']}" if raw.get("unit") else ""
        fields = [f"{label}={value}{unit}"]
        if raw.get("status") is not None:
            fields.append(f"status={_compact(raw['status'])}")
        if raw.get("normal_range") is not None:
            fields.append(f"normal_range={_compact(raw['normal_range'])}")
        if raw.get("message"):
            fields.append(f"note={_compact(raw['message'])}")
        lines.append("- " + "; ".join(fields))
    return lines


def build_diagnostic_context(
    snapshot: Mapping[str, Any],
    *,
    application_name: str,
    max_chars: int = 6000,
    stale_after_seconds: int = 120,
    allow_paths: Iterable[str] = (),
    deny_patterns: Iterable[str] = DEFAULT_DENY_PATTERNS,
    now: datetime | None = None,
) -> str:
    """Build bounded plain-text context for the local LLM.

    A richer ``parameters`` mapping is rendered with labels, units, ranges, and
    status. Any other JSON shape is flattened automatically.
    """
    now = now or datetime.now(timezone.utc)
    source = snapshot.get("source", application_name)
    timestamp = snapshot.get("timestamp")
    captured_at = _parse_timestamp(timestamp)

    lines = [
        f"application: {_compact(application_name)}",
        f"source: {_compact(source)}",
        f"application_status: {_compact(snapshot.get('status', 'unknown'))}",
    ]

    if captured_at:
        age_seconds = max(0, int((now - captured_at).total_seconds()))
        freshness = "stale" if age_seconds > stale_after_seconds else "fresh"
        lines.append(
            f"snapshot_time_utc: {captured_at.isoformat()} age_seconds={age_seconds} freshness={freshness}"
        )
    else:
        lines.append("snapshot_time_utc: unavailable freshness=unknown")

    parameters = snapshot.get("parameters")
    if isinstance(parameters, Mapping):
        lines.append("parameters:")
        lines.extend(
            _structured_parameter_lines(
                parameters,
                allow_paths=allow_paths,
                deny_patterns=deny_patterns,
            )
            or ["- none"]
        )
    else:
        excluded_top_level = {"source", "timestamp", "status", "events"}
        arbitrary = {k: v for k, v in snapshot.items() if k not in excluded_top_level}
        lines.append("parameters:")
        flattened = _flatten(
            arbitrary,
            allow_paths=allow_paths,
            deny_patterns=deny_patterns,
        )
        lines.extend((f"- {path}: {_compact(value)}" for path, value in flattened))
        if not flattened:
            lines.append("- none")

    events = snapshot.get("events")
    if isinstance(events, list) and events:
        lines.append("recent_events:")
        for event in events[-10:]:
            if isinstance(event, Mapping):
                severity = _compact(event.get("severity", "info"))
                message = _compact(event.get("message", event))
                lines.append(f"- severity={severity}; message={message}")
            else:
                lines.append(f"- {_compact(event)}")

    context = "\n".join(lines)
    if len(context) > max_chars:
        context = context[: max(0, max_chars - 24)] + "\n[context truncated]"
    return context
