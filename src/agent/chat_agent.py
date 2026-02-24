"""
Main chat agent — orchestrates the full pipeline.

Pipeline:
  message → session check → greeting/company flow → company data fetch
  → Q&A cache check → classify → SQL generate → SQL execute
  → web research (if needed) → compose answer → cache → return
"""
from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd

from src.agent.session_manager import get_session_manager, SessionState
from src.cache.company_cache import get_company_cache
from src.cache.query_cache import get_query_cache
from src.data.supabase_loader import get_supabase_loader
from src.data.data_cleaner import clean_dataframe
from src.data.schema_analyzer import analyze_schema
from src.pipeline.question_classifier import classify_question
from src.pipeline.sql_generator import generate_sql
from src.pipeline.sql_executor import execute_multiple, merge_results, build_empty_hint, SQLExecutionError
from src.pipeline.web_researcher import research_topic, format_web_context
from src.pipeline.answer_composer import compose_answer
from src.utils.logger import get_logger
from src.utils.text_utils import sanitize_company_name

logger = get_logger(__name__)

# Greeting keywords that should trigger company name prompt
_GREETING_KEYWORDS = {
    "hi", "hello", "hey", "good morning", "good afternoon",
    "good evening", "howdy", "greetings", "sup", "what's up",
}


@dataclass
class ChatResponse:
    session_id: str
    response: str
    question_type: str
    company: Optional[str]
    from_cache: bool
    pipeline_steps: list[str]
    error: Optional[str] = None


async def handle_message(session_id: str, message: str) -> ChatResponse:
    """
    Main entry point. Process a user message end-to-end.
    """
    steps: list[str] = []
    session_manager = get_session_manager()
    session = session_manager.get_or_create(session_id)

    # ── Step 1: Greeting / company collection flow ───────────────────────────

    # Pure greeting → ask for company
    if _is_greeting(message) and not session.company_name:
        session.awaiting_company = True
        return ChatResponse(
            session_id=session_id,
            response=(
                "Hello! I'm your Employee Survey Insights assistant. "
                "Which company's data would you like to explore?"
            ),
            question_type="greeting",
            company=None,
            from_cache=False,
            pipeline_steps=["greeting"],
        )

    # Awaiting company name → treat message as company name
    if session.awaiting_company and not session.company_name:
        company_raw = message.strip()
        return await _set_company_and_respond(
            session, session_id, company_raw, steps
        )

    # No company set yet and not a greeting → prompt for company
    if not session.company_name:
        # Check if message itself could be a company name (short, no question words)
        if _looks_like_company_name(message):
            return await _set_company_and_respond(session, session_id, message, steps)

        session.awaiting_company = True
        return ChatResponse(
            session_id=session_id,
            response="Which company's survey data would you like to analyze?",
            question_type="prompt_company",
            company=None,
            from_cache=False,
            pipeline_steps=["prompt_company"],
        )

    # ── Detect company switch ───────────────────────────────────────────────
    switched = await _detect_company_switch(session, message)
    if switched:
        steps.append("company_switch")
        get_query_cache().clear_session(session_id)

    # ── Step 2: Load company data ───────────────────────────────────────────
    steps.append("load_data")
    df, metadata = await _ensure_company_data(session)
    if df is None:
        return ChatResponse(
            session_id=session_id,
            response=metadata.get("error", "Failed to load company data."),
            question_type="error",
            company=session.company_name,
            from_cache=False,
            pipeline_steps=steps,
            error=metadata.get("error"),
        )

    # ── Q&A Cache check ──────────────────────────────────────────────────────
    steps.append("cache_check")
    query_cache = get_query_cache()
    cached = query_cache.get(session_id, message)
    if cached:
        _, cached_response = cached
        return ChatResponse(
            session_id=session_id,
            response=cached_response,
            question_type="cached",
            company=session.company_name,
            from_cache=True,
            pipeline_steps=steps + ["cache_hit"],
        )

    # ── Step 3: Classify question ────────────────────────────────────────────
    steps.append("classify")
    try:
        classification = await classify_question(message, metadata)
    except Exception as e:
        logger.error("classify_error", error=str(e))
        classification = {
            "question_type": "simple",
            "intent": message,
            "requires_web_search": False,
            "aggregation": None,
            "dimensions": [],
            "metric": None,
            "filters": [],
            "complexity": "simple",
        }

    question_type = classification.get("question_type", "simple")

    # ── Step 4: SQL Generation ───────────────────────────────────────────────
    steps.append("sql_generate")
    result_df = pd.DataFrame()
    hint_msg = ""

    try:
        sql_queries = await generate_sql(message, classification, metadata, df)

        # ── Step 5: SQL Execution ────────────────────────────────────────────
        steps.append("sql_execute")
        result_dfs = execute_multiple(sql_queries, df)
        result_df = merge_results(result_dfs)

        if result_df.empty and sql_queries:
            hint_msg = build_empty_hint(sql_queries[0], df, classification, metadata)

    except SQLExecutionError as e:
        logger.error("sql_execution_failed", error=str(e))
        hint_msg = str(e)
    except Exception as e:
        logger.error("sql_pipeline_error", error=str(e))
        hint_msg = "I encountered an issue processing your query."

    # ── Step 6: Web Research ─────────────────────────────────────────────────
    web_context = ""
    if classification.get("requires_web_search") and session.company_name:
        steps.append("web_research")
        try:
            web_results = await research_topic(
                company_name=session.company_name,
                question=message,
                question_type=question_type,
                classification=classification,
            )
            web_context = format_web_context(web_results)
        except Exception as e:
            logger.warning("web_research_failed", error=str(e))

    # ── Step 7: Compose Answer ───────────────────────────────────────────────
    steps.append("compose")

    if result_df.empty and hint_msg:
        response = hint_msg
    else:
        try:
            response = await compose_answer(
                question=message,
                classification=classification,
                result_df=result_df,
                web_context=web_context,
                conversation_history=session.get_history_dicts(),
                company_name=session.company_name or "",
            )
        except Exception as e:
            logger.error("compose_answer_error", error=str(e))
            response = (
                "I'm sorry, I encountered an error composing the answer. "
                "Please try rephrasing your question."
            )

    # ── Step 8: Cache & Return ───────────────────────────────────────────────
    if not result_df.empty:
        query_cache.set(
            session_id=session_id,
            question=message,
            result_df=result_df,
            response=response,
        )

    session.add_turn(question=message, answer=response, question_type=question_type)

    return ChatResponse(
        session_id=session_id,
        response=response,
        question_type=question_type,
        company=session.company_name,
        from_cache=False,
        pipeline_steps=steps,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_greeting(message: str) -> bool:
    import re

    msg = message.lower().strip()
    if msg in _GREETING_KEYWORDS:
        return True

    single_word_greetings = {g for g in _GREETING_KEYWORDS if " " not in g}
    words = set(re.findall(r"[a-z']+", msg))
    return bool(words & single_word_greetings) and len(words) <= 5


def _looks_like_company_name(message: str) -> bool:
    """Heuristic: short message with no analytical intent keywords."""
    q_words = {"what", "which", "how", "who", "when", "where", "why", "show", "give", "list"}
    analytics_words = {
        "average", "avg", "mean", "sum", "count", "total", "top", "bottom",
        "highest", "lowest", "rank", "ranking", "compare", "comparison",
        "difference", "trend", "distribution", "insight", "insights",
        "recommendation", "recommendations", "score", "scores", "metric",
        "metrics", "department", "departments", "employee", "employees",
        "response", "responses", "question", "questions", "by", "over", "time",
    }
    words = message.lower().split()
    if "?" in message:
        return False
    if len(words) == 0 or len(words) > 4:
        return False
    blocked = q_words | analytics_words
    return not any(w in blocked for w in words)


async def _set_company_and_respond(
    session: SessionState,
    session_id: str,
    company_raw: str,
    steps: list[str],
) -> ChatResponse:
    """Set company name in session, trigger data load, return confirmation."""
    steps.append("set_company")
    company_sanitized = sanitize_company_name(company_raw)
    session.switch_company(company_sanitized)

    df, metadata = await _ensure_company_data(session)
    if df is None:
        return ChatResponse(
            session_id=session_id,
            response=metadata.get(
                "error",
                f"I couldn't find data for '{company_raw}'. "
                "Please check the company name and try again.",
            ),
            question_type="error",
            company=company_sanitized,
            from_cache=False,
            pipeline_steps=steps,
            error=metadata.get("error"),
        )

    table_summary = metadata.get("table_summary", "employee survey data")
    row_count = len(df)
    response = (
        f"Got it! I've loaded {row_count:,} survey records for **{company_raw}**. "
        f"Dataset: {table_summary}\n\n"
        "What would you like to know?"
    )

    return ChatResponse(
        session_id=session_id,
        response=response,
        question_type="company_set",
        company=company_sanitized,
        from_cache=False,
        pipeline_steps=steps,
    )


async def _ensure_company_data(
    session: SessionState,
) -> tuple[Optional[pd.DataFrame], dict]:
    """
    Return (df, metadata) from cache or fresh fetch + analyze.
    On error returns (None, {"error": "..."}).
    """
    company_name = session.company_name
    if not company_name:
        return None, {"error": "No company selected."}

    company_cache = get_company_cache()

    cached = company_cache.get(company_name)
    if cached:
        return cached  # (df, metadata)

    # Fetch fresh data
    loader = get_supabase_loader()
    try:
        raw_df = loader.fetch_company_data(company_name)
    except ValueError as e:
        return None, {"error": str(e)}
    except Exception as e:
        logger.error("supabase_fetch_error", company=company_name, error=str(e))
        return None, {"error": f"Failed to load data for '{company_name}': {e}"}

    # Clean
    df = clean_dataframe(raw_df)

    # Analyze schema (LLM)
    metadata = await analyze_schema(df)

    # Cache
    company_cache.set(company_name, df, metadata)

    return df, metadata


async def _detect_company_switch(session: SessionState, message: str) -> bool:
    """
    Detect if the user is asking about a different company.
    Returns True if a switch was made.
    """
    msg_lower = message.lower()
    switch_phrases = ["switch to", "change to", "now for", "for company", "about company"]
    if not any(phrase in msg_lower for phrase in switch_phrases):
        return False

    # Try to extract the new company name from the message
    import re
    patterns = [
        r"switch(?:ing)? to (.+?)(?:\?|$)",
        r"change(?:ing)? to (.+?)(?:\?|$)",
        r"(?:now for|for company|about company)\s+(.+?)(?:\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            new_company = match.group(1).strip()
            new_company_sanitized = sanitize_company_name(new_company)
            if new_company_sanitized != session.company_name:
                session.switch_company(new_company_sanitized)
                return True
    return False
