import json
from typing import Any, Optional
import anthropic
from src.config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.text_utils import extract_json_block, strip_markdown_fences

logger = get_logger(__name__)


class LLMClient:
    """Unified async Anthropic LLM wrapper with streaming support."""

    def __init__(self):
        settings = get_settings()
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.llm_model

    async def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """Send a completion request and return the text response.

        Uses streaming internally to avoid timeouts on long responses.
        """
        try:
            async with self._client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                temperature=temperature,
                **kwargs,
            ) as stream:
                final = await stream.get_final_message()

            for block in final.content:
                if block.type == "text":
                    return block.text

            return ""

        except anthropic.RateLimitError as e:
            logger.warning("llm_rate_limit", error=str(e))
            raise
        except anthropic.APIConnectionError as e:
            logger.error("llm_connection_error", error=str(e))
            raise
        except anthropic.APIStatusError as e:
            logger.error("llm_api_error", status=e.status_code, error=str(e))
            raise

    async def complete_json(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> Optional[Any]:
        """Complete and parse the response as JSON.

        Returns the parsed object, or None on failure.
        """
        text = await self.complete(
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        text = strip_markdown_fences(text)

        parsed = _safe_json_loads(text)
        if parsed is not None:
            return parsed

        # Attempt to extract first JSON object/array from mixed text.
        block = extract_json_block(text)
        if block:
            parsed = _safe_json_loads(block)
            if parsed is not None:
                return parsed

        # Final repair pass for minor truncation/brace imbalance.
        repaired = _repair_json_candidate(block or text)
        parsed = _safe_json_loads(repaired) if repaired else None
        if parsed is not None:
            logger.info("llm_json_parse_recovered")
            return parsed

        logger.warning("llm_json_parse_failed", text_preview=text[:200])
        return None

    async def classify(
        self,
        system: str,
        user_content: str,
        max_tokens: int = 512,
    ) -> Optional[Any]:
        """Lightweight classification call returning JSON."""
        return await self.complete_json(
            system=system,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=max_tokens,
            temperature=0.0,
        )


_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def _safe_json_loads(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _repair_json_candidate(text: str) -> str:
    """
    Best-effort fix for small JSON issues:
    - trim leading/trailing prose around first JSON token
    - append missing closing braces/brackets
    """
    if not text:
        return ""

    candidate = text.strip()

    start_positions = [p for p in (candidate.find("{"), candidate.find("[")) if p >= 0]
    if not start_positions:
        return candidate

    start = min(start_positions)
    candidate = candidate[start:]

    # Trim to likely JSON end.
    end_obj = candidate.rfind("}")
    end_arr = candidate.rfind("]")
    end = max(end_obj, end_arr)
    if end >= 0:
        candidate = candidate[: end + 1]

    # Balance braces/brackets if response was truncated.
    open_obj = candidate.count("{")
    close_obj = candidate.count("}")
    if open_obj > close_obj:
        candidate += "}" * (open_obj - close_obj)

    open_arr = candidate.count("[")
    close_arr = candidate.count("]")
    if open_arr > close_arr:
        candidate += "]" * (open_arr - close_arr)

    return candidate
