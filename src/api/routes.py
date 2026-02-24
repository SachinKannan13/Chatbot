import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from src.agent.chat_agent import handle_message
from src.agent.session_manager import get_session_manager
from src.cache.company_cache import get_company_cache
from src.cache.query_cache import get_query_cache
from src.config.settings import get_settings
from src.data.chat_history_store import get_chat_history_store
from src.pipeline.relatable_fallback_engine import get_relatable_engine
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        examples=["user-abc-123"],
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        examples=["hi"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"session_id": "user-abc-123", "message": "hi"},
                {"session_id": "user-abc-123", "message": "What is the average score by department?"},
            ]
        }
    }


class ChatResponse(BaseModel):
    session_id: str
    response: str
    question_type: str
    company: Optional[str]
    from_cache: bool
    pipeline_steps: list[str]
    error: Optional[str] = None


class SessionResetRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        examples=["user-abc-123"],
    )


class SessionResetResponse(BaseModel):
    session_id: str
    status: str


class HealthResponse(BaseModel):
    status: str
    version: str
    active_sessions: int
    cached_companies: list[str]


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return a response."""
    try:
        relatable_engine = get_relatable_engine()

        # If user replied "No" to fallback, move to next candidate.
        # Any other text exits fallback and continues normal pipeline.
        confirmation = await relatable_engine.handle_confirmation(
            session_id=request.session_id,
            user_message=request.message,
            normal_handler=handle_message,
        )
        if confirmation.handled:
            return ChatResponse(
                session_id=request.session_id,
                response=confirmation.response,
                question_type=confirmation.question_type,
                company=get_session_manager().get_or_create(request.session_id).company_name,
                from_cache=False,
                pipeline_steps=["relatable_fallback"],
                error=None,
            )

        result = await handle_message(
            session_id=request.session_id,
            message=request.message,
        )

        # Strict trigger: fallback runs only if normal response indicates no data.
        fallback = await relatable_engine.maybe_run(
            session_id=request.session_id,
            user_query=request.message,
            base_result=result,
            normal_response_text=result.response,
            normal_handler=handle_message,
        )
        if fallback.handled:
            return ChatResponse(
                session_id=request.session_id,
                response=fallback.response,
                question_type=fallback.question_type,
                company=result.company,
                from_cache=False,
                pipeline_steps=result.pipeline_steps + ["relatable_fallback"],
                error=None,
            )

        # Persist only real user Q&A turns (skip onboarding/company setup messages).
        non_qa_types = {"greeting", "prompt_company", "company_set"}
        if result.question_type not in non_qa_types:
            await asyncio.to_thread(
                get_chat_history_store().save_turn,
                question=request.message,
                answer=result.response,
                category=result.question_type,
            )
        return ChatResponse(
            session_id=result.session_id,
            response=result.response,
            question_type=result.question_type,
            company=result.company,
            from_cache=result.from_cache,
            pipeline_steps=result.pipeline_steps,
            error=result.error,
        )
    except ValueError as e:
        logger.warning("chat_request_error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("chat_unhandled_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again.",
        )


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    settings = get_settings()
    session_mgr = get_session_manager()
    company_cache = get_company_cache()

    return HealthResponse(
        status="ok",
        version=settings.api_version,
        active_sessions=session_mgr.active_count(),
        cached_companies=list(company_cache.info().keys()),
    )


@router.post("/session/reset", response_model=SessionResetResponse)
async def reset_session(request: SessionResetRequest):
    """Clear session state and Q&A cache for a session."""
    session_mgr = get_session_manager()
    query_cache = get_query_cache()

    session_mgr.reset(request.session_id)
    query_cache.clear_session(request.session_id)

    logger.info("session_reset_via_api", session_id=request.session_id)
    return SessionResetResponse(
        session_id=request.session_id,
        status="reset",
    )


@router.get("/cache/info")
async def cache_info():
    """Debug endpoint: show cache status."""
    settings = get_settings()
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Not found")

    company_cache = get_company_cache()
    query_cache = get_query_cache()
    return {
        "company_cache": company_cache.info(),
        "query_cache_size": query_cache.size(),
    }
