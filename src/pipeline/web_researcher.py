import asyncio
from src.utils.logger import get_logger
from src.config.settings import get_settings

logger = get_logger(__name__)


class WebSearchResult:
    def __init__(self, title: str, url: str, content: str):
        self.title = title
        self.url = url
        self.content = content

    def __str__(self) -> str:
        return f"[{self.title}]\n{self.content[:500]}"


async def research_topic(
    company_name: str,
    question: str,
    question_type: str,
    classification: dict,
) -> list[WebSearchResult]:
    """
    Perform web search for analytics/insights/recommendations questions.

    Returns a list of search results (top 3).
    Returns empty list if web search is not configured or fails.
    """
    settings = get_settings()
    if not settings.web_search_api_key:
        logger.info("web_search_skipped_no_key")
        return []

    queries = _build_queries(company_name, question, question_type, classification)
    results = []

    for query in queries[:2]:  # Limit to 2 queries to control latency
        try:
            batch = await _tavily_search(query, settings.web_search_api_key)
            results.extend(batch)
        except Exception as e:
            logger.warning("web_search_error", query=query, error=str(e))

    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for r in results:
        if r.url not in seen_urls:
            seen_urls.add(r.url)
            unique_results.append(r)
        if len(unique_results) >= 3:
            break

    logger.info("web_research_complete", results_count=len(unique_results))
    return unique_results


def _build_queries(
    company_name: str,
    question: str,
    question_type: str,
    classification: dict,
) -> list[str]:
    """Build targeted search queries based on the question type and intent."""
    intent = classification.get("intent", question)
    queries = []

    if question_type == "recommendations":
        queries.append(f"how to improve employee engagement {_extract_topic(intent)}")
        queries.append(f"best practices {_extract_topic(intent)} employee experience")

    elif question_type == "insights":
        queries.append(f"employee survey insights {_extract_topic(intent)}")
        queries.append(f"what drives {_extract_topic(intent)} employee satisfaction")

    elif question_type == "analytics":
        queries.append(f"employee {_extract_topic(intent)} analytics patterns")
        queries.append(f"{company_name} employee engagement strategy")

    else:
        queries.append(f"employee survey {_extract_topic(intent)} best practices")

    return queries


def _extract_topic(intent: str) -> str:
    """Extract key topic words from the intent."""
    import re
    # Remove common stop words and return key terms
    stop_words = {
        "what", "is", "the", "are", "of", "for", "in", "and", "or", "a", "an",
        "which", "department", "company", "employee", "survey", "data", "show",
        "tell", "me", "give", "find", "get",
    }
    words = re.findall(r"\b[a-zA-Z]+\b", intent.lower())
    key_words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(key_words[:5])


async def _tavily_search(
    query: str, api_key: str, max_results: int = 3
) -> list[WebSearchResult]:
    """Perform a Tavily search and return results."""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = await asyncio.to_thread(
            client.search,
            query=query,
            max_results=max_results,
            search_depth="basic",
        )
        results = []
        for item in (response.get("results") or []):
            results.append(
                WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                )
            )
        return results
    except ImportError:
        logger.warning("tavily_not_installed")
        return []


def format_web_context(results: list[WebSearchResult]) -> str:
    """Format search results into a context string for the LLM."""
    if not results:
        return ""
    lines = ["Web research context:\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"Source {i}: {r.title}")
        lines.append(r.content[:600])
        lines.append("")
    return "\n".join(lines)
