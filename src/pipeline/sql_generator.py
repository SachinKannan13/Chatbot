import difflib
from typing import Any, Optional
import pandas as pd
from src.utils.llm_client import get_llm_client
from src.utils.logger import get_logger
from src.utils.text_utils import strip_markdown_fences

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are an expert SQL generator for SQLite.

The data is loaded into an in-memory SQLite table called "dataset".
Always use exact column names in double quotes: "ColumnName".

STRICT RULES:
1. Table name is always: dataset
2. Always quote column names in double quotes: "ColumnName"
3. SCORE / METRIC QUESTIONS — always use AVG, never SUM or raw values:
   AVG(CAST("Score" AS REAL)) AS avg_score
4. For sums: SUM(CAST("ColumnName" AS REAL))
5. For ranking: ORDER BY avg_score DESC/ASC + LIMIT N
6. For comparisons: use CASE WHEN pivot queries
7. For multi-intent: separate queries with ---SQL_BREAK---
8. NEVER use column names not in the provided schema
9. For date grouping: strftime('%Y-%m', "date_col") AS month
10. Always end with semicolon
11. STRING FILTERS — use the EXACT value from the "Exact distinct values" list provided.
    Use COLLATE NOCASE for safety: WHERE "col" = 'Value' COLLATE NOCASE
12. For "highest/best/top": ORDER BY avg_score DESC
    For "lowest/worst/bottom": ORDER BY avg_score ASC

SCORE AVERAGING RULE (critical):
When a user asks about a score (highest score, lowest score, average score, score by X):
- ALWAYS GROUP BY the dimension and compute AVG per group
- NEVER return raw individual Score rows
- Pattern: SELECT "Dim", AVG(CAST("Score" AS REAL)) AS avg_score FROM dataset GROUP BY "Dim" ORDER BY avg_score DESC

Return ONLY the SQL query. No explanation, no markdown.
For multi-intent, use ---SQL_BREAK--- between each query."""


async def generate_sql(
    question: str,
    classification: dict[str, Any],
    metadata: dict[str, Any],
    df: pd.DataFrame,
) -> list[str]:
    """
    Generate SQL query(ies) for the question.

    Returns a list of SQL strings (one per intent for multi-intent questions).
    Validates column names and retries once on failure.
    """
    columns = df.columns.tolist()
    question_type = classification.get("question_type", "simple")

    prompt = _build_prompt(question, classification, metadata, df, columns)

    llm = get_llm_client()
    sql_text = await llm.complete(
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.0,
    )

    sql_text = strip_markdown_fences(sql_text)

    # Split multi-intent
    queries = [q.strip() for q in sql_text.split("---SQL_BREAK---") if q.strip()]

    # Validate each query
    validated = []
    for sql in queries:
        ok, error = _validate_columns(sql, columns)
        if ok:
            validated.append(sql)
        else:
            logger.warning("sql_column_validation_failed", error=error, sql=sql[:200])
            retry_sql = await _retry_with_error(question, sql, error, metadata, df, columns)
            if retry_sql:
                validated.append(retry_sql)
            else:
                fallback = _rule_based_sql(question_type, classification, metadata, columns)
                if fallback:
                    validated.append(fallback)

    if not validated:
        validated = ["SELECT * FROM dataset LIMIT 100;"]

    return validated


def _build_prompt(
    question: str,
    classification: dict,
    metadata: dict,
    df: pd.DataFrame,
    columns: list[str],
) -> str:
    primary_metric = metadata.get("primary_metric_column", "")
    primary_dims = metadata.get("primary_dimension_columns", [])
    date_col = metadata.get("date_column")
    col_roles = metadata.get("column_roles", {})
    question_type = classification.get("question_type", "simple")
    object_nunique = df.select_dtypes(include=["object"]).nunique(dropna=True).to_dict()

    # Sample rows (3 rows)
    sample_rows = df.head(3).to_dict(orient="records")
    sample_str = "\n".join(str(row) for row in sample_rows)

    # Column role summary + EXACT distinct values for categorical columns
    role_lines = []
    for col in columns:
        info = col_roles.get(col, {})
        col_type = info.get("type", "unknown")
        role = info.get("role", "unknown")
        is_score = info.get("is_score", False)

        if col_type in ("categorical", "identifier") or (
            df[col].dtype == object and object_nunique.get(col, 0) < 200
        ):
            # Include exact distinct values — critical for correct WHERE clauses
            exact_vals = sorted(df[col].dropna().unique().astype(str).tolist())[:30]
            role_lines.append(
                f"  '{col}': {col_type}/{role}, is_score={is_score}, "
                f"EXACT VALUES: {exact_vals}"
            )
        elif col_type == "numeric" or is_score:
            try:
                col_num = pd.to_numeric(df[col], errors="coerce")
                role_lines.append(
                    f"  '{col}': numeric/metric, is_score={is_score}, "
                    f"range=[{col_num.min():.2f}, {col_num.max():.2f}], "
                    f"mean={col_num.mean():.2f}"
                )
            except Exception:
                role_lines.append(f"  '{col}': numeric/metric, is_score={is_score}")
        else:
            role_lines.append(f"  '{col}': {col_type}/{role}, is_score={is_score}")

    role_summary = "\n".join(role_lines)

    prompt = f"""Table name: dataset
Available columns: {', '.join(f'"{c}"' for c in columns)}
Primary metric/score column: "{primary_metric}"
Primary dimension columns: {primary_dims}
Date column: {f'"{date_col}"' if date_col else 'null'}

Column details (use EXACT VALUES listed for WHERE filters):
{role_summary}

Classification:
- Question type: {question_type}
- Intent: {classification.get('intent', question)}
- Aggregation: {classification.get('aggregation', 'none')}
- Dimensions needed: {classification.get('dimensions', [])}
- Filters: {classification.get('filters', [])}

Sample rows:
{sample_str}

User question: {question}

IMPORTANT REMINDERS:
- Score questions → always GROUP BY dimension, use AVG(CAST("{primary_metric}" AS REAL)) AS avg_score
- String WHERE values → use EXACT values from the list above, COLLATE NOCASE for safety
- Table is always "dataset", column names always in double quotes"""

    # Type-specific instructions
    if question_type in ("aggregation", "calculation", "simple"):
        if primary_metric:
            prompt += (
                f"\n\nFor score/metric questions: "
                f'SELECT "dim_col", AVG(CAST("{primary_metric}" AS REAL)) AS avg_score '
                f"FROM dataset GROUP BY \"dim_col\" ORDER BY avg_score DESC;"
            )
    elif question_type == "ranking":
        prompt += f'\n\nRanking: GROUP BY dimension, AVG score, ORDER BY avg_score DESC/ASC, LIMIT 10.'
    elif question_type == "comparison":
        prompt += "\n\nComparison: use CASE WHEN pivot to show values side-by-side with gap column."
    elif question_type == "multi_intent":
        prompt += "\n\nMulti-intent: generate separate SQL queries separated by ---SQL_BREAK---."
    elif question_type == "trend":
        prompt += f"\n\nTrend: GROUP BY date using strftime('%Y-%m', \"{date_col}\") AS month."
    elif question_type == "single_intent":
        prompt += "\n\nDeep analysis: fetch all rows for the relevant filter with SELECT *."
    elif question_type in ("insights", "analytics", "recommendations"):
        prompt += (
            f"\n\nFor insights/recommendations: return aggregated scores grouped by all "
            f"relevant dimensions. Include AVG, MIN, MAX, COUNT per group."
        )

    return prompt


def _validate_columns(sql: str, valid_columns: list[str]) -> tuple[bool, str]:
    """Check that all double-quoted identifiers in SQL exist in the schema."""
    import re
    quoted = re.findall(r'"([^"]+)"', sql)
    quoted = [q for q in quoted if q.lower() != "dataset"]

    for col in quoted:
        if col not in valid_columns:
            close = difflib.get_close_matches(col, valid_columns, n=1, cutoff=0.75)
            suggestion = f" Did you mean '{close[0]}'?" if close else ""
            return False, f"Column '{col}' not found in schema.{suggestion}"

    return True, ""


async def _retry_with_error(
    question: str,
    bad_sql: str,
    error: str,
    metadata: dict,
    df: pd.DataFrame,
    columns: list[str],
) -> Optional[str]:
    """Retry SQL generation with error context."""
    # Build exact value list for retry
    value_hints = []
    for col in df.select_dtypes(include=["object"]).columns:
        vals = sorted(df[col].dropna().unique().astype(str).tolist())[:15]
        value_hints.append(f"  '{col}': {vals}")

    llm = get_llm_client()
    prompt = f"""The following SQL has an error:

{bad_sql}

Error: {error}

Available columns (exact, double-quoted): {', '.join(f'"{c}"' for c in columns)}

Exact string values per column:
{chr(10).join(value_hints)}

Fix the SQL. Return ONLY the corrected SQL query, no explanation."""

    result = await llm.complete(
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.0,
    )
    result = strip_markdown_fences(result)

    ok, _ = _validate_columns(result, columns)
    return result if ok else None


def _rule_based_sql(
    question_type: str,
    classification: dict,
    metadata: dict,
    columns: list[str],
) -> Optional[str]:
    """Fallback rule-based SQL for common patterns."""
    primary_metric = metadata.get("primary_metric_column")
    primary_dims = metadata.get("primary_dimension_columns", [])

    if primary_metric and primary_metric not in columns:
        primary_metric = None
    valid_dims = [d for d in primary_dims if d in columns]

    if question_type in ("aggregation", "ranking", "calculation", "simple",
                          "insights", "analytics", "recommendations"):
        if primary_metric and valid_dims:
            dim = valid_dims[0]
            order = "ASC" if any(
                k in classification.get("intent", "").lower()
                for k in ["lowest", "worst", "bottom", "minimum"]
            ) else "DESC"
            limit = "LIMIT 10" if question_type == "ranking" else ""
            return (
                f'SELECT "{dim}", '
                f'AVG(CAST("{primary_metric}" AS REAL)) AS avg_score, '
                f'MIN(CAST("{primary_metric}" AS REAL)) AS min_score, '
                f'MAX(CAST("{primary_metric}" AS REAL)) AS max_score, '
                f'COUNT(*) AS response_count '
                f'FROM dataset '
                f'GROUP BY "{dim}" '
                f'ORDER BY avg_score {order} '
                f'{limit};'
            ).strip()
        elif primary_metric:
            return (
                f'SELECT AVG(CAST("{primary_metric}" AS REAL)) AS avg_score, '
                f'MIN(CAST("{primary_metric}" AS REAL)) AS min_score, '
                f'MAX(CAST("{primary_metric}" AS REAL)) AS max_score, '
                f'COUNT(*) AS response_count '
                f'FROM dataset;'
            )

    elif question_type == "count":
        return "SELECT COUNT(*) AS total_count FROM dataset;"

    elif question_type == "list":
        dims = classification.get("dimensions", valid_dims)
        if dims and dims[0] in columns:
            return f'SELECT DISTINCT "{dims[0]}" FROM dataset ORDER BY "{dims[0]}";'

    return "SELECT * FROM dataset LIMIT 50;"
