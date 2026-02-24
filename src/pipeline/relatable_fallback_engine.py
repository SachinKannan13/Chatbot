"""
RelatableFallbackEngine — handles queries that reference a missing dataset value.

Finds similar alternatives, substitutes the best match into the original query,
and lets the user step through ranked options ("no" → next, "yes"/"ok" → accept).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from threading import Lock
from typing import Any, Awaitable, Callable, Optional

import pandas as pd

from src.agent.session_manager import get_session_manager
from src.cache.company_cache import get_company_cache
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

_YES_WORDS = {
    "yes", "y", "yeah", "yep", "yup", "ok", "okay", "sure",
    "fine", "correct", "right", "confirmed", "confirm",
}
_NO_WORDS = {
    "no", "n", "nope", "nah", "next", "skip",
    "wrong", "incorrect", "not this", "not that",
}

# Question types that should never trigger the fallback
_NON_TRIGGER_TYPES = {"greeting", "prompt_company", "company_set", "error", "cached"}

# Substrings in a pipeline response that signal "no useful data was returned".
# Keep this list TIGHT — loose phrases like "couldn't find" or "please refine"
# can appear inside legitimate LLM answers and would cause false-positive triggers.
# These two cover every message produced by build_empty_hint() and
# answer_composer._deterministic_answer() when result_df is empty.
_NO_DATA_MARKERS = (
    "no data found",
    "no data was found",
)

_STATE_TTL_MINUTES    = 15    # inactive fallback state expires after this
_MAX_CANDIDATES       = 8     # maximum ranked alternatives to keep
_SIMILARITY_THRESHOLD = 0.15  # minimum score to be considered a candidate

# Values that are never useful as fallback suggestions
_NOISE_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"  # UUID
    r"|^[^\s@]+@[^\s@]+\.[^\s@]+$"                                        # email
    r"|^\d+$",                                                             # pure integer
    re.IGNORECASE,
)

# Common query words that are never the "missing value" the user meant
_STOP_WORDS = {
    "what", "which", "how", "who", "when", "where", "why", "is", "are",
    "the", "a", "an", "for", "of", "in", "on", "at", "by", "to", "and",
    "or", "show", "give", "list", "tell", "me", "please", "all", "any",
    "average", "avg", "mean", "sum", "count", "total", "score", "scores",
    "top", "bottom", "best", "worst", "highest", "lowest", "number",
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RelatableOutcome:
    handled: bool
    response: str = ""
    question_type: str = "relatable_fallback"


@dataclass
class _FallbackState:
    session_id: str
    original_query: str
    missing_value: str
    source_column: str                  # column where the value was expected
    candidates: list[dict[str, Any]]    # ranked: [{"value", "column", "similarity"}, ...]
    index: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def expired(self) -> bool:
        return datetime.utcnow() > self.created_at + timedelta(minutes=_STATE_TTL_MINUTES)


# ── Engine ────────────────────────────────────────────────────────────────────

class RelatableFallbackEngine:
    """
    Handles queries that reference a value not present in the company dataset.

    Fully general — works for any column type (grades, departments, statements,
    health levers, custom codes, etc.) without any hardcoded type mappings.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._states: dict[str, _FallbackState] = {}

    # ── Public interface ───────────────────────────────────────────────────────

    async def handle_confirmation(
        self,
        session_id: str,
        user_message: str,
        normal_handler: Callable[[str, str], Awaitable[Any]],
    ) -> RelatableOutcome:
        """
        Call this BEFORE the normal pipeline on every message.

        If a fallback state is active for this session:
          - "no"          → present the next ranked candidate.
          - "yes" / "ok"  → acknowledge and clear fallback state.
          - anything else → clear state; return handled=False so the normal
                            pipeline processes the user's new question.
        """
        state = self._get_state(session_id)
        if state is None:
            return RelatableOutcome(handled=False)

        word = (user_message or "").strip().lower()

        # ── "no" → advance to next candidate ──────────────────────────────────
        if word in _NO_WORDS:
            state.index += 1
            if state.index >= len(state.candidates):
                self._clear_state(session_id)
                return RelatableOutcome(
                    handled=True,
                    response=(
                        f'No more alternatives found for **"{state.missing_value}"**. '
                        "Please try rephrasing your question."
                    ),
                )
            return await self._present(state, normal_handler)

        # ── "yes" / "ok" → acknowledge and close ──────────────────────────────
        if word in _YES_WORDS:
            self._clear_state(session_id)
            return RelatableOutcome(
                handled=True,
                response="Got it! Feel free to ask your next question.",
                question_type="relatable_confirmed",
            )

        # ── Any other text → new question; let normal pipeline handle it ───────
        self._clear_state(session_id)
        return RelatableOutcome(handled=False)

    async def maybe_run(
        self,
        session_id: str,
        user_query: str,
        base_result: Any,
        normal_response_text: str,
        normal_handler: Callable[[str, str], Awaitable[Any]],
    ) -> RelatableOutcome:
        """
        Call this AFTER the normal pipeline completes.

        Triggers only when:
          - The question type is not an onboarding/setup type.
          - The pipeline response indicates no useful data was found.
          - A specific missing value can be extracted.
          - That value genuinely does not exist in the dataset.
          - At least one similar candidate is found.
        """
        qtype = getattr(base_result, "question_type", "") or ""
        if qtype in _NON_TRIGGER_TYPES:
            return RelatableOutcome(handled=False)

        if not self._is_no_data_response(normal_response_text):
            return RelatableOutcome(handled=False)

        # Load cached company DataFrame
        session = get_session_manager().get_or_create(session_id)
        if not session.company_name:
            return RelatableOutcome(handled=False)

        cached = get_company_cache().get(session.company_name)
        if not cached:
            return RelatableOutcome(handled=False)
        df, _metadata = cached

        # Extract (missing_value, source_column)
        missing_value, source_column = self._extract_missing_info(
            normal_response_text, user_query, df
        )
        if not missing_value:
            return RelatableOutcome(handled=False)

        # If the value actually exists, don't trigger (avoids false positives)
        if self._value_exists(missing_value, source_column, df):
            return RelatableOutcome(handled=False)

        # Find similar candidates in the data
        candidates = self._find_candidates(missing_value, source_column, df)
        if not candidates:
            return RelatableOutcome(handled=False)

        logger.info(
            "relatable_fallback_triggered",
            missing=missing_value,
            source_column=source_column,
            candidates_found=len(candidates),
        )

        state = _FallbackState(
            session_id=session_id,
            original_query=user_query,
            missing_value=missing_value,
            source_column=source_column,
            candidates=candidates,
        )
        self._set_state(state)
        return await self._present(state, normal_handler)

    # ── Core presentation ──────────────────────────────────────────────────────

    async def _present(
        self,
        state: _FallbackState,
        normal_handler: Optional[Callable[[str, str], Awaitable[Any]]],
    ) -> RelatableOutcome:
        """
        Substitute the rank-N candidate into the original query, run it through
        the full pipeline, and wrap the real result in a fallback response.
        """
        if normal_handler is None or state.index >= len(state.candidates):
            self._clear_state(state.session_id)
            return RelatableOutcome(
                handled=True,
                response=f'No alternatives found for **"{state.missing_value}"**.',
            )

        candidate  = state.candidates[state.index]
        substitute = candidate["value"]
        col        = candidate["column"]
        sim        = candidate["similarity"]
        rank       = state.index + 1
        total      = len(state.candidates)

        # Snapshot history so the substitute query doesn't pollute conversation context.
        session = get_session_manager().get_or_create(state.session_id)
        history_snapshot = len(session.conversation_history)

        rewritten   = self._rewrite_query(state.original_query, state.missing_value, substitute)
        answer_obj  = await normal_handler(state.session_id, rewritten)
        result_text = getattr(answer_obj, "response", "") or "No data was found."

        # Remove the turn that handle_message added for the substitute query
        session.conversation_history = session.conversation_history[:history_snapshot]

        header = (
            f'**"{state.missing_value}"** was not found in the dataset.\n\n'
            f'Closest match #{rank}: **"{substitute}"**'
            + (f" _(column: {col})_" if col else "")
            + f" — {sim:.0%} similarity\n\n"
        )
        footer = (
            f'\n\n---\nReply **"no"** to see alternative #{rank + 1} of {total}.'
            if rank < total
            else "\n\n---\nThis was the last available alternative."
        )

        return RelatableOutcome(
            handled=True,
            response=header + result_text + footer,
            question_type="relatable_fallback",
        )

    # ── No-data detection ──────────────────────────────────────────────────────

    def _is_no_data_response(self, text: str) -> bool:
        """Return True if the pipeline response indicates no useful data was returned."""
        t = (text or "").strip().lower()
        if not t:
            return True
        if any(marker in t for marker in _NO_DATA_MARKERS):
            return True
        # Catch deterministic fallback output where every value is N/A:
        # e.g. "1. avg_score: N/A" or "Results (1 rows):\n  avg_score: N/A"
        if ":" in t:
            rhs_values = []
            for line in t.splitlines():
                if ":" not in line:
                    continue
                _, rhs = line.split(":", 1)
                val = re.sub(r"^\d+\.\s*", "", rhs.strip().lower().strip(".,"))
                if val:
                    rhs_values.append(val)
            if rhs_values and all(v in {"n/a", "na", "none", "null", "nan", "-"} for v in rhs_values):
                return True
        return False

    # ── Missing-value extraction ───────────────────────────────────────────────

    def _extract_missing_info(
        self,
        hint_text: str,
        user_query: str,
        df: pd.DataFrame,
    ) -> tuple[str, str]:
        """Return (missing_value, source_column) via hint parsing or query scan."""
        # Strategy 1: parse the hint message
        m = re.search(
            r"[Nn]o data found for (.+?) = ['\"]([^'\"]{2,})['\"]",
            hint_text,
        )
        if m:
            col = m.group(1).strip()
            val = m.group(2).strip()
            if val and not self._is_noise(val):
                return val, col

        # Strategy 2: parse the user query
        return self._find_missing_in_query(user_query, df)

    def _find_missing_in_query(self, query: str, df: pd.DataFrame) -> tuple[str, str]:
        """
        Scan the user query for a token that does NOT appear in any string
        column of df. Returns (missing_value, best_matching_column).
        """
        # Build a flat lookup: lowercased value → column name
        known: dict[str, str] = {}
        for col in df.select_dtypes(include=["object"]).columns:
            for v in df[col].dropna().unique():
                vs = str(v).strip()
                if not self._is_noise(vs):
                    known[vs.lower()] = col

        # Column-name tokens — phrases containing these are context labels, not
        # missing entities (e.g. "grade" before "JJK" → skip "Grade JJK", keep "JJK").
        col_name_tokens: set[str] = set()
        for col in df.select_dtypes(include=["object"]).columns:
            for tok in re.findall(r"[a-z0-9]+", col.lower()):
                col_name_tokens.add(tok)

        # Collect candidate phrases: quoted strings first, then n-grams
        candidates: list[str] = []

        for m in re.finditer(r'"([^"]{2,})"' + r"|'([^']{2,})'", query):
            candidates.append((m.group(1) or m.group(2)).strip())

        if not candidates:
            tokens = re.findall(r"[A-Za-z0-9]+(?:[.\-_][A-Za-z0-9]+)*", query)
            # Bigrams first, then single tokens; skip stop words and column-name labels.
            for size in (2, 1):
                for i in range(len(tokens) - size + 1):
                    phrase = " ".join(tokens[i: i + size])
                    phrase_toks = [p.lower() for p in phrase.split()]
                    if any(p in _STOP_WORDS for p in phrase_toks):
                        continue
                    if any(p in col_name_tokens for p in phrase_toks):
                        continue
                    if len(phrase) >= 2:
                        candidates.append(phrase)

        # Return the first candidate that is not in the data
        for cand in candidates:
            if cand.lower() not in known and not self._is_noise(cand):
                best_col = self._infer_best_column(cand, query, df)
                return cand, best_col

        return "", ""

    def _infer_best_column(self, missing_value: str, query: str, df: pd.DataFrame) -> str:
        """
        Infer which column the missing value belongs to.
        A: closest column-name token appearing BEFORE the value in the query.
        B: column whose name has most token overlap with the full query.
        C: column name most similar to the missing value (last resort).
        """
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if not obj_cols:
            return ""

        query_lower = query.lower()
        mv_pos = query_lower.find(missing_value.lower())

        # Strategy A: column whose name token appears closest BEFORE the
        # missing value in the query text (word-boundary match only).
        if mv_pos > 0:
            best_col = ""
            best_dist = float("inf")
            for col in obj_cols:
                col_tokens = re.findall(r"[a-z0-9]+", col.lower())
                for token in col_tokens:
                    pattern = r"\b" + re.escape(token) + r"\b"
                    for m in re.finditer(pattern, query_lower):
                        if m.start() < mv_pos:
                            dist = mv_pos - m.start()
                            if dist < best_dist:
                                best_dist = dist
                                best_col = col
            if best_col:
                return best_col

        # Strategy B: column name token overlap with the full query
        query_tokens = set(re.findall(r"[a-z0-9]+", query_lower))
        best_overlap_col = ""
        best_overlap = 0
        for col in obj_cols:
            col_tokens = set(re.findall(r"[a-z0-9]+", col.lower()))
            overlap = len(col_tokens & query_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_overlap_col = col
        if best_overlap_col:
            return best_overlap_col

        # Strategy C: column name most similar to the missing value
        best_col, best_sim = "", 0.0
        for col in obj_cols:
            s = SequenceMatcher(None, missing_value.lower(), col.lower()).ratio()
            if s > best_sim:
                best_sim, best_col = s, col
        return best_col

    def _value_exists(self, value: str, column: str, df: pd.DataFrame) -> bool:
        """Return True if `value` (case-insensitive) already exists in the data."""
        v_low = value.lower()
        cols = (
            [column]
            if (column and column in df.columns)
            else df.select_dtypes(include=["object"]).columns.tolist()
        )
        for col in cols:
            if v_low in {str(x).lower() for x in df[col].dropna().unique()}:
                return True
        return False

    # ── Candidate search ───────────────────────────────────────────────────────

    def _find_candidates(
        self,
        missing_value: str,
        source_column: str,
        df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Find values in the DataFrame that are similar to `missing_value`.

        When source_column is known (from the hint message), ONLY that column
        is searched — we never cross into unrelated columns like Employee Name.
        This is the key rule: if the user asked about a grade and the grade
        column is identified, alternatives come exclusively from that column.

        When source_column is unknown, all columns are searched with a threshold.
        """
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()

        def _score_column(col: str) -> list[dict[str, Any]]:
            results = []
            for val in df[col].dropna().unique():
                vs = str(val).strip()
                if vs.lower() == missing_value.lower() or self._is_noise(vs):
                    continue
                sim = self._similarity(missing_value, vs)
                results.append({"value": vs, "column": col, "similarity": sim})
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results

        # Source column known: search only that column, no threshold.
        if source_column and source_column in obj_cols:
            from_source = _score_column(source_column)
            return from_source[:_MAX_CANDIDATES]

        # ── Source column unknown: search all, apply threshold ─────────────────
        seen: set[str] = set()
        scored: list[dict[str, Any]] = []
        for col in obj_cols:
            for item in _score_column(col):
                key = item["value"].lower()
                if key not in seen:
                    seen.add(key)
                    scored.append(item)

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        filtered = [c for c in scored if c["similarity"] >= _SIMILARITY_THRESHOLD]
        return filtered[:_MAX_CANDIDATES]

    # ── Query rewriting ────────────────────────────────────────────────────────

    def _rewrite_query(self, original: str, missing: str, substitute: str) -> str:
        """Replace `missing` with `substitute` in the original query (case-insensitive)."""
        if missing and missing.lower() in original.lower():
            return re.compile(re.escape(missing), re.IGNORECASE).sub(
                substitute, original, count=1
            )
        return f"{original} (use {substitute} instead)"

    # ── Similarity scoring ─────────────────────────────────────────────────────

    def _similarity(self, a: str, b: str) -> float:
        """
        Fuzzy similarity score in [0, 1] via rapidfuzz.

        For short strings (≤5 chars, e.g. codes like "JKK", "L2") plain
        character-level ratio is used because token-based matching is noisy
        on such short inputs.  For longer strings WRatio combines edit
        distance, partial-ratio, and token-set matching automatically.
        """
        from rapidfuzz import fuzz
        fn = fuzz.ratio if max(len(a), len(b)) <= 5 else fuzz.WRatio
        return round(fn(a, b) / 100.0, 4)

    # ── Noise filter ───────────────────────────────────────────────────────────

    def _is_noise(self, value: str) -> bool:
        """Return True for values that are never useful as fallback candidates."""
        v = (value or "").strip()
        return not v or len(v) <= 1 or bool(_NOISE_RE.search(v))

    # ── Thread-safe state management ───────────────────────────────────────────

    def _get_state(self, session_id: str) -> Optional[_FallbackState]:
        with self._lock:
            s = self._states.get(session_id)
            if s is None:
                return None
            if s.expired():
                del self._states[session_id]
                return None
            return s

    def _set_state(self, state: _FallbackState) -> None:
        with self._lock:
            self._states[state.session_id] = state

    def _clear_state(self, session_id: str) -> None:
        with self._lock:
            self._states.pop(session_id, None)


# ── Singleton ──────────────────────────────────────────────────────────────────

_engine: Optional[RelatableFallbackEngine] = None


def get_relatable_engine() -> RelatableFallbackEngine:
    global _engine
    if _engine is None:
        _engine = RelatableFallbackEngine()
    return _engine
