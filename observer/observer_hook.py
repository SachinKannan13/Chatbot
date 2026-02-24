from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Any

from observer.logger import summarize_supabase_response
from observer.report_writer import write_report
from observer import cognitive_trace


_LOCK = threading.Lock()
_STATE: dict[str, Any] = {
    "user_query": "",
    "supabase": {
        "raw_response": {},
        "response_type": "",
        "row_count": 0,
    },
    "llm": {
        "input": "",
        "output": "",
        "model": "",
    },
}
_PATCHED = False


def _enabled() -> bool:
    return os.getenv("OBSERVER_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_user_query(messages: list[dict[str, Any]] | None) -> str:
    if not messages:
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            return str(content)
    return str(messages[-1].get("content", "")) if isinstance(messages[-1], dict) else ""


def _capture_supabase(response: Any) -> None:
    try:
        snapshot = summarize_supabase_response(response)
        with _LOCK:
            _STATE["supabase"] = snapshot
    except Exception:
        pass


def _capture_llm_input(system: str, messages: list[dict[str, Any]], model: str) -> None:
    try:
        llm_input = f"SYSTEM:\n{system}\n\nMESSAGES:\n{messages}"
        with _LOCK:
            _STATE["user_query"] = _extract_user_query(messages)
            _STATE["llm"] = {
                "input": llm_input,
                "output": _STATE.get("llm", {}).get("output", ""),
                "model": model or "",
            }
    except Exception:
        pass


def _write_report_with_output(output: str) -> None:
    try:
        with _LOCK:
            report = {
                "timestamp": _utc_now(),
                "user_query": _STATE.get("user_query", ""),
                "supabase": dict(_STATE.get("supabase", {})),
                "llm": {
                    "input": _STATE.get("llm", {}).get("input", ""),
                    "output": output,
                    "model": _STATE.get("llm", {}).get("model", ""),
                },
            }
            _STATE["llm"]["output"] = output
        try:
            cognitive_trace.analyze(
                supabase_response=report.get("supabase", {}),
                llm_input=report.get("llm", {}).get("input", ""),
                llm_output=output,
                report_object=report,
            )
        except Exception:
            pass
        write_report(report)
    except Exception:
        pass


def _patch_supabase_execute() -> None:
    _SUFFIXES = (
        "SelectRequestBuilder", "FilterRequestBuilder", "QueryRequestBuilder",
        "SingleRequestBuilder", "MaybeSingleRequestBuilder", "RPCFilterRequestBuilder",
    )

    def _sync_wrapper(fn):
        def wrapper(self, *args, **kwargs):
            response = fn(self, *args, **kwargs)
            _capture_supabase(response)
            return response
        wrapper._observer_patched = True  # type: ignore[attr-defined]
        return wrapper

    def _async_wrapper(fn):
        async def wrapper(self, *args, **kwargs):
            response = await fn(self, *args, **kwargs)
            _capture_supabase(response)
            return response
        wrapper._observer_patched = True  # type: ignore[attr-defined]
        return wrapper

    def _patch_module(rb_module, prefix: str, make_wrapper) -> None:
        for suffix in _SUFFIXES:
            cls = getattr(rb_module, f"{prefix}{suffix}", None)
            if cls is None or not hasattr(cls, "execute"):
                continue
            if getattr(cls.execute, "_observer_patched", False):
                continue
            cls.execute = make_wrapper(cls.execute)

    try:
        from postgrest._sync import request_builder as sync_rb
        _patch_module(sync_rb, "Sync", _sync_wrapper)
    except Exception:
        pass

    try:
        from postgrest._async import request_builder as async_rb
        _patch_module(async_rb, "Async", _async_wrapper)
    except Exception:
        pass


def _patch_llm_client() -> None:
    try:
        from src.utils.llm_client import LLMClient

        if getattr(LLMClient.complete, "_observer_patched", False):
            return

        original_complete = LLMClient.complete

        async def patched_complete(
            self,
            system: str,
            messages: list[dict],
            max_tokens: int = 2048,
            temperature: float = 0.0,
            **kwargs,
        ) -> str:
            try:
                _capture_llm_input(system, messages, getattr(self, "model", ""))
            except Exception:
                pass

            output = await original_complete(
                self,
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            try:
                _write_report_with_output(output)
            except Exception:
                pass

            return output

        patched_complete._observer_patched = True  # type: ignore[attr-defined]
        LLMClient.complete = patched_complete
    except Exception:
        pass


def attach_observer() -> None:
    global _PATCHED
    if _PATCHED or not _enabled():
        return
    _patch_supabase_execute()
    _patch_llm_client()
    _PATCHED = True


attach_observer()
