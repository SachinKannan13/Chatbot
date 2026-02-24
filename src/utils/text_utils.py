import re
import unicodedata
from typing import Optional


_FENCE_PREFIX = "```"


def normalize_question(text: str) -> str:
    """Lowercase, strip, collapse whitespace, remove punctuation."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def sanitize_company_name(name: str) -> str:
    """Convert company name to Supabase table-safe format."""
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    return name


def extract_numbers_from_text(text: str) -> list[float]:
    """
    Extract numeric values (int/float/comma-formatted/percent) from text.
    """
    pattern = r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|\b\d+(?:\.\d+)?%?\b"
    matches = re.findall(pattern, text)
    results = []
    for m in matches:
        clean = m.rstrip("%").replace(",", "")
        try:
            results.append(float(clean))
        except ValueError:
            pass
    return results


def extract_json_block(text: str) -> Optional[str]:
    """Extract the first JSON object or array block from text."""
    match = re.search(r"\{[\s\S]*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    match = re.search(r"\[[\s\S]*\]", text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def strip_markdown_fences(text: str) -> str:
    """Strip a single markdown code fence wrapper if present."""
    text = (text or "").strip()
    if not text.startswith(_FENCE_PREFIX):
        return text
    lines = text.split("\n")
    lines = lines[1:]
    if lines and lines[-1].strip().startswith(_FENCE_PREFIX):
        lines = lines[:-1]
    return "\n".join(lines).strip()
