from typing import Any
import pandas as pd
from src.utils.llm_client import get_llm_client
from src.utils.logger import get_logger
from src.data.data_cleaner import get_column_type_summary

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a data analyst. Analyze the provided dataset schema and return a JSON object describing column roles.

Return ONLY valid JSON — no markdown, no explanation.

Required format:
{
  "column_roles": {
    "<ColumnName>": {
      "type": "categorical|numeric|date|identifier",
      "role": "dimension|metric|identifier|date",
      "description": "What this column represents",
      "is_score": true/false,
      "is_primary_metric": true/false,
      "semantic_aliases": ["alias1", "alias2"]
    }
  },
  "primary_metric_column": "<ColumnName or null>",
  "primary_dimension_columns": ["col1", "col2"],
  "date_column": "<ColumnName or null>",
  "table_summary": "One sentence describing the dataset"
}"""


async def analyze_schema(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run a one-time LLM deep analysis of the DataFrame schema.
    Returns structured metadata used in all subsequent SQL generation.
    """
    column_types = get_column_type_summary(df)
    column_stats = _build_column_stats(df, column_types)

    prompt = _build_prompt(df.columns.tolist(), column_stats)

    llm = get_llm_client()
    result = await llm.complete_json(
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000,
        temperature=0.0,
    )

    if result is None:
        logger.warning("schema_analysis_failed_llm", columns=df.columns.tolist())
        result = _build_fallback_metadata(df, column_types)

    if result.get("primary_metric_column") is None and "Score" in df.columns:
        result["primary_metric_column"] = "Score"

    logger.info("schema_analyzed", primary_metric=result.get("primary_metric_column"))
    return result


def _build_column_stats(df: pd.DataFrame, column_types: dict) -> dict:
    """Build per-column statistics for the LLM prompt."""
    stats = {}
    for col in df.columns:
        col_type = column_types.get(col, "categorical")
        col_stats: dict = {"type": col_type}

        if col_type == "numeric":
            col_data = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(col_data) > 0:
                col_stats["min"] = round(float(col_data.min()), 4)
                col_stats["max"] = round(float(col_data.max()), 4)
                col_stats["mean"] = round(float(col_data.mean()), 4)
                col_stats["non_null_count"] = int(col_data.count())

        elif col_type in ("categorical", "identifier"):
            sample_values = df[col].dropna().unique()[:20].tolist()
            # Convert to strings for safe serialization
            col_stats["sample_values"] = [str(v) for v in sample_values]
            col_stats["unique_count"] = int(df[col].nunique(dropna=True))

        elif col_type == "date":
            col_stats["sample_values"] = df[col].dropna().astype(str).unique()[:5].tolist()

        stats[col] = col_stats
    return stats


def _build_prompt(columns: list[str], column_stats: dict) -> str:
    lines = ["Dataset columns and statistics:\n"]
    for col in columns:
        stats = column_stats.get(col, {})
        col_type = stats.get("type", "unknown")

        if col_type == "numeric":
            lines.append(
                f"- '{col}' (numeric): min={stats.get('min')}, "
                f"max={stats.get('max')}, mean={stats.get('mean')}"
            )
        elif col_type in ("categorical", "identifier"):
            samples = stats.get("sample_values", [])[:10]
            lines.append(
                f"- '{col}' ({col_type}): {stats.get('unique_count', 0)} unique values, "
                f"samples: {samples}"
            )
        elif col_type == "date":
            samples = stats.get("sample_values", [])
            lines.append(f"- '{col}' (date): samples: {samples}")
        else:
            lines.append(f"- '{col}' ({col_type})")

    return "\n".join(lines)


def _build_fallback_metadata(df: pd.DataFrame, column_types: dict) -> dict:
    """Build a minimal metadata dict without LLM when analysis fails."""
    numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
    categorical_cols = [c for c, t in column_types.items() if t == "categorical"]
    date_cols = [c for c, t in column_types.items() if t == "date"]

    # Guess primary metric (look for 'score'-like columns)
    primary_metric = None
    for col in numeric_cols:
        if any(kw in col.lower() for kw in ["score", "rating", "value", "index"]):
            primary_metric = col
            break
    if primary_metric is None and numeric_cols:
        primary_metric = numeric_cols[0]

    column_roles = {}
    for col in df.columns:
        col_type = column_types.get(col, "categorical")
        column_roles[col] = {
            "type": col_type,
            "role": "metric" if col_type == "numeric" else
                    "date" if col_type == "date" else
                    "identifier" if col_type == "identifier" else "dimension",
            "description": f"{col} column",
            "is_score": col == primary_metric,
            "is_primary_metric": col == primary_metric,
            "semantic_aliases": [col.lower().replace("_", " ")],
        }

    return {
        "column_roles": column_roles,
        "primary_metric_column": primary_metric,
        "primary_dimension_columns": categorical_cols[:3],
        "date_column": date_cols[0] if date_cols else None,
        "table_summary": f"Dataset with {len(df.columns)} columns and {len(df)} rows.",
    }
