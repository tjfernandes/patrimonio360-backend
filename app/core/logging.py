from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
from typing import Any

from app.core.config import Settings

_REQUEST_ID = contextvars.ContextVar("request_id", default="-")
_CONFIGURED = False


def get_request_id() -> str:
    return _REQUEST_ID.get()


def set_request_id(request_id: str) -> contextvars.Token[str]:
    return _REQUEST_ID.set(request_id or "-")


def reset_request_id(token: contextvars.Token[str]) -> None:
    _REQUEST_ID.reset(token)


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = get_request_id()
        return True


def configure_backend_logging(settings: Settings) -> None:
    """Configure terminal logs for operational backend events.

    This is intentionally separate from QueryLogger JSONL evaluation logs.
    """

    global _CONFIGURED
    if _CONFIGURED or not settings.BACKEND_LOG_ENABLED:
        return

    level = getattr(logging, settings.BACKEND_LOG_LEVEL.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s [%(name)s] [rid=%(request_id)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    app_logger = logging.getLogger("app")
    app_logger.handlers.clear()
    app_logger.addHandler(handler)
    app_logger.setLevel(level)
    app_logger.propagate = False

    _CONFIGURED = True


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return f"list(len={len(value)})"
    if isinstance(value, dict):
        keys = ",".join(str(key) for key in list(value.keys())[:8])
        suffix = ",..." if len(value) > 8 else ""
        return f"dict(len={len(value)},keys={keys}{suffix})"

    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 180:
        text = f"{text[:177]}..."
    if not text:
        return '""'
    if re.search(r"[\s=]", text):
        return repr(text)
    return text


def format_kv(**fields: Any) -> str:
    return " ".join(
        f"{key}={_format_value(value)}"
        for key, value in fields.items()
        if value is not None
    )


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: Any,
) -> None:
    if not logger.isEnabledFor(level):
        return
    suffix = format_kv(**fields)
    logger.log(level, "%s%s", event, f" {suffix}" if suffix else "")


def _is_numeric_sequence(values: list[Any]) -> bool:
    return bool(values) and all(isinstance(value, (int, float)) for value in values)


def sanitize_for_terminal_json(value: Any, *, max_string_chars: int = 1200) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, bytes):
        return {"_type": "bytes", "length": len(value)}
    if isinstance(value, str):
        if len(value) <= max_string_chars:
            return value
        return {
            "_type": "truncated_string",
            "length": len(value),
            "preview": value[:max_string_chars],
        }
    if hasattr(value, "model_dump"):
        try:
            return sanitize_for_terminal_json(value.model_dump(mode="json"))
        except Exception:
            pass
    if isinstance(value, dict):
        return {
            str(key): sanitize_for_terminal_json(item, max_string_chars=max_string_chars)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if _is_numeric_sequence(value):
            return {
                "_type": "vector",
                "dimension": len(value),
                "head": [round(float(item), 6) for item in value[:8]],
            }
        if value and all(isinstance(item, list) and _is_numeric_sequence(item) for item in value):
            first = value[0]
            return {
                "_type": "vector_batch",
                "count": len(value),
                "dimension": len(first),
                "first_head": [round(float(item), 6) for item in first[:8]],
            }
        return [sanitize_for_terminal_json(item, max_string_chars=max_string_chars) for item in value]
    return str(value)


def log_json_event(
    logger: logging.Logger,
    level: int,
    event: str,
    payload: Any,
    *,
    max_chars: int = 40000,
) -> None:
    if not logger.isEnabledFor(level):
        return
    sanitized = sanitize_for_terminal_json(payload)
    try:
        rendered = json.dumps(sanitized, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    except Exception:
        rendered = json.dumps(str(sanitized), ensure_ascii=False, indent=2)
    if max_chars > 0 and len(rendered) > max_chars:
        rendered = (
            f"{rendered[:max_chars]}\n"
            f"... truncated terminal JSON: {len(rendered) - max_chars} chars omitted ..."
        )
    logger.log(level, "%s\n%s", event, rendered)
