import re
import sqlite3
from typing import Any, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SQLExecutionError(Exception):
    pass


def _build_string_value_lookup(df: pd.DataFrame) -> dict[str, str]:
    value_lookup: dict[str, str] = {}
    for col in df.select_dtypes(include=["object"]).columns:
        for val in df[col].dropna().unique():
            if isinstance(val, str) and val.strip():
                value_lookup[val.lower().strip()] = val
    return value_lookup


def resolve_sql_string_values(
    sql: str,
    df: pd.DataFrame,
    value_lookup: Optional[dict[str, str]] = None,
) -> str:
    """
    Post-process generated SQL to fix string literal casing mismatches.

    For every quoted string literal 'value' in the SQL, look up the closest
    matching actual value in any string column of the DataFrame (case-insensitive).
    Replaces with the exact-case value found in the data.
    """
    if value_lookup is None:
        value_lookup = _build_string_value_lookup(df)

    if not value_lookup:
        return sql

    def replace_literal(match: re.Match) -> str:
        raw_literal = match.group(1)
        literal = raw_literal.replace("''", "'")
        exact = value_lookup.get(literal.lower().strip())
        if exact and exact != literal:
            escaped = exact.replace("'", "''")
            logger.debug("sql_value_case_resolved", original=literal, resolved=exact)
            return f"'{escaped}'"
        return match.group(0)

    # Replace SQL single-quoted literals while preserving escaped apostrophes.
    return re.sub(r"'((?:''|[^'])*)'", replace_literal, sql)


def _execute_on_connection(sql: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """Execute one SQL query against a prepared SQLite connection."""
    try:
        result_df = pd.read_sql_query(sql, conn)
        logger.debug(
            "sql_executed",
            rows_returned=len(result_df),
            columns=result_df.columns.tolist(),
        )
        return result_df
    except (sqlite3.Error, pd.errors.DatabaseError) as e:
        logger.warning("sql_execution_error", error=str(e), sql=sql[:300])
        raise SQLExecutionError(str(e)) from e


def execute_sql(sql: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute a SQL query on an in-memory SQLite instance loaded with df.

    Raises SQLExecutionError on failure.
    """
    value_lookup = _build_string_value_lookup(df)
    prepared_sql = resolve_sql_string_values(sql, df, value_lookup)

    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA case_sensitive_like = OFF;")
        conn.row_factory = sqlite3.Row
        df.to_sql("dataset", conn, index=False, if_exists="replace")
        return _execute_on_connection(prepared_sql, conn)
    finally:
        if conn:
            conn.close()


def execute_multiple(sqls: list[str], df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Execute multiple SQL queries on the same DataFrame.

    Returns a list of result DataFrames (one per query).
    Failed queries are skipped and logged.
    """
    results: list[pd.DataFrame] = []
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA case_sensitive_like = OFF;")
        conn.row_factory = sqlite3.Row
        df.to_sql("dataset", conn, index=False, if_exists="replace")
        value_lookup = _build_string_value_lookup(df)

        for i, sql in enumerate(sqls):
            try:
                prepared_sql = resolve_sql_string_values(sql, df, value_lookup)
                results.append(_execute_on_connection(prepared_sql, conn))
            except SQLExecutionError as e:
                logger.warning("multi_sql_query_failed", index=i, error=str(e))
                results.append(pd.DataFrame())
        return results
    finally:
        if conn:
            conn.close()


def build_empty_hint(
    sql: str,
    df: pd.DataFrame,
    classification: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    """
    When SQL returns an empty result, build a helpful hint message listing
    the actual available values for the filtered column.
    """
    where_pattern = re.findall(r'"([^"]+)"\s*=\s*[\'\"]([^\'\"]+)[\'\"]', sql, re.IGNORECASE)
    hints = []

    for col, value in where_pattern:
        if col in df.columns:
            available = df[col].dropna().unique().tolist()[:20]
            available_str = ", ".join(f"'{v}'" for v in available)
            hints.append(
                f"No data found for {col} = '{value}'. "
                f"Available values for '{col}': {available_str}"
            )

    if hints:
        return " | ".join(hints)

    dims = metadata.get("primary_dimension_columns", [])
    if dims and dims[0] in df.columns:
        col = dims[0]
        available = df[col].dropna().unique().tolist()[:10]
        return (
            f"No data found matching your filters. "
            f"Available '{col}' values: {', '.join(str(v) for v in available)}"
        )

    return "No data found matching your query. Please refine your question."


def merge_results(results: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple result DataFrames into one for multi-intent questions."""
    non_empty = [r for r in results if not r.empty]
    if not non_empty:
        return pd.DataFrame()
    if len(non_empty) == 1:
        return non_empty[0]
    return pd.concat(non_empty, axis=0, ignore_index=True)
