import functools
import hashlib
from typing import Optional
import pandas as pd
from cachetools import TTLCache
from src.utils.logger import get_logger
from src.utils.text_utils import normalize_question

logger = get_logger(__name__)


def _make_cache_key(session_id: str, question: str) -> str:
    normalized = normalize_question(question)
    raw = f"{session_id}::{normalized}"
    return hashlib.sha256(raw.encode()).hexdigest()


class QueryCache:
    """
    Per-session Q&A cache.

    Key: hash(session_id + normalized_question)
    Constraint check: cached result must satisfy the same filters as the new question.
    """

    def __init__(self, ttl_hours: int = 2):
        self._store: TTLCache[str, tuple[pd.DataFrame, str]] = TTLCache(
            maxsize=100000,
            ttl=ttl_hours * 3600,
        )

    def get(
        self,
        session_id: str,
        question: str,
        constraint_check: Optional[dict] = None,
    ) -> Optional[tuple[pd.DataFrame, str]]:
        """
        Return (result_df, response) if cached, not expired, and constraints match.

        constraint_check: optional dict with keys like {"filters": {...}}
        used to detect if cached result satisfies tighter filters than before.
        """
        key = _make_cache_key(session_id, question)
        entry = self._store.get(key)
        if entry is None:
            return None
        result_df, response = entry

        if constraint_check and not self._constraints_satisfied(result_df, constraint_check):
            logger.debug("query_cache_constraint_mismatch", key=key[:8])
            return None

        logger.debug("query_cache_hit", key=key[:8])
        return result_df, response

    def set(
        self,
        session_id: str,
        question: str,
        result_df: pd.DataFrame,
        response: str,
    ) -> str:
        key = _make_cache_key(session_id, question)
        self._store[key] = (result_df, response)
        logger.debug("query_cache_set", key=key[:8])
        return key

    def _constraints_satisfied(self, result_df: pd.DataFrame, constraint_check: dict) -> bool:
        """
        Check if the cached DataFrame satisfies constraint filters.
        """
        filters: dict = constraint_check.get("filters", {})
        for col, _ in filters.items():
            if col not in result_df.columns:
                return False
            unique_vals = result_df[col].dropna().unique()
            if len(unique_vals) > 1:
                return False
        return True

    def clear(self) -> None:
        self._store.clear()
        logger.info("query_cache_cleared")

    def size(self) -> int:
        return len(self._store)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)


class SessionQueryCache:
    """
    Wraps QueryCache with session-aware key tracking for O(1) session clears.
    """

    def __init__(self, ttl_hours: int = 2):
        self._cache = QueryCache(ttl_hours=ttl_hours)
        self._session_keys: dict[str, set[str]] = {}

    def get(
        self,
        session_id: str,
        question: str,
        constraint_check: Optional[dict] = None,
    ) -> Optional[tuple[pd.DataFrame, str]]:
        return self._cache.get(session_id, question, constraint_check)

    def set(
        self,
        session_id: str,
        question: str,
        result_df: pd.DataFrame,
        response: str,
    ) -> None:
        key = self._cache.set(session_id, question, result_df, response)
        if session_id not in self._session_keys:
            self._session_keys[session_id] = set()
        self._session_keys[session_id].add(key)

    def clear_session(self, session_id: str) -> None:
        keys = self._session_keys.pop(session_id, set())
        for key in keys:
            self._cache.delete(key)
        logger.info("session_query_cache_cleared", session_id=session_id, keys_removed=len(keys))

    def clear_all(self) -> None:
        self._cache.clear()
        self._session_keys.clear()

    def size(self) -> int:
        return self._cache.size()


@functools.cache
def get_query_cache() -> SessionQueryCache:
    from src.config.settings import get_settings
    return SessionQueryCache(ttl_hours=get_settings().query_cache_ttl_hours)
