from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _filename_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%fZ")


def write_report(payload: dict[str, Any]) -> str | None:
    """
    Write a single report JSON file. Returns file path on success.
    Failures are swallowed to keep observer non-intrusive.
    """
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = dict(payload)
        payload.setdefault("timestamp", _timestamp_now())
        filename = REPORTS_DIR / f"report_{_filename_timestamp()}.json"
        with filename.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return str(filename)
    except Exception:
        return None

