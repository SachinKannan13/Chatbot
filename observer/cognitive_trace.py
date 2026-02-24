from __future__ import annotations

import ast
import math
import re
from statistics import mean, pvariance
from typing import Any

from observer import derivation_trace

def analyze(
    supabase_response: Any,
    llm_input: str,
    llm_output: str,
    report_object: dict[str, Any],
) -> None:
    """
    Passively extend report_object with a cognitive_trace section.
    This function is exception-safe and non-intrusive by design.
    """
    try:
        rows = derivation_trace._extract_rows(supabase_response)
        structural = _structural_analysis(supabase_response, rows)
        semantic = _semantic_analysis(structural, llm_input, llm_output)
        prompt = _prompt_analysis(llm_input, supabase_response)
        metric = _metric_reconstruction(rows, structural)
        transform = _transformation_detection(llm_output, metric)
        justification = _justification_mapping(llm_output, metric)
        reasoning = _reasoning_trace(structural, semantic, transform, metric)
        derivations = derivation_trace.analyze_derivations(
            supabase_response=supabase_response,
            llm_input=llm_input,
            llm_output=llm_output,
            structural_analysis=structural,
            semantic_analysis=semantic,
            metric_reconstruction=metric,
            transformation_detection=transform,
        )

        report_object["cognitive_trace"] = {
            "structural_analysis": structural,
            "semantic_analysis": semantic,
            "prompt_analysis": prompt,
            "transformation_detection": transform,
            "metric_reconstruction": metric,
            "justification_mapping": justification,
            "reasoning_trace": reasoning,
            "derivation_trace": derivations,
        }
    except Exception as e:
        report_object["cognitive_trace"] = {
            "error": f"cognitive_trace_failed: {e}",
            "structural_analysis": {},
            "semantic_analysis": {},
            "prompt_analysis": {},
            "transformation_detection": {"operations_detected": []},
            "metric_reconstruction": {},
            "justification_mapping": {"mappings": []},
            "reasoning_trace": {"steps": []},
            "derivation_trace": {
                "mathematical_operations": [],
                "formulas_detected": [],
                "formula_reconstructions": [],
                "logical_derivations": [],
                "inference_rules": [],
                "derivation_confidence": [],
            },
        }


def _structural_analysis(supabase_response: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    response_type = type(supabase_response).__name__
    if isinstance(supabase_response, dict) and "response_type" in supabase_response:
        response_type = str(supabase_response.get("response_type", response_type))

    columns = _collect_columns(rows)
    numeric_fields: list[str] = []
    categorical_fields: list[str] = []
    for col in columns:
        vals = [r.get(col) for r in rows if col in r and r.get(col) is not None]
        if vals and _is_numeric_series(vals):
            numeric_fields.append(col)
        else:
            categorical_fields.append(col)

    return {
        "response_type": response_type,
        "row_count": len(rows),
        "columns": columns,
        "numeric_fields": numeric_fields,
        "categorical_fields": categorical_fields,
    }


def _semantic_analysis(
    structural: dict[str, Any],
    llm_input: str,
    llm_output: str,
) -> dict[str, Any]:
    text = f"{llm_input}\n{llm_output}".lower()
    cols = [c.lower() for c in structural.get("columns", [])]
    numeric_cols = [c.lower() for c in structural.get("numeric_fields", [])]

    metric_type = "unknown"
    if any("percent" in c or "%" in c for c in cols):
        metric_type = "percentage"
    elif any("score" in c for c in numeric_cols):
        metric_type = "score"
    elif any("rating" in c for c in numeric_cols):
        metric_type = "rating"
    elif any("count" in c for c in cols):
        metric_type = "count"
    elif numeric_cols:
        metric_type = "numeric"

    domain = "unknown"
    if any(k in text for k in ("employee", "survey", "engagement", "department", "health lever")):
        domain = "employee engagement"
    elif any(k in text for k in ("performance", "productivity")):
        domain = "performance"
    elif any(k in text for k in ("finance", "revenue", "profit", "cost")):
        domain = "finance"

    aggregation_level = "record"
    for candidate in ("department", "category", "company", "team", "region", "quarter", "month", "year"):
        if candidate in text or any(candidate in c for c in cols):
            aggregation_level = candidate
            break

    return {
        "metric_type": metric_type,
        "domain": domain,
        "aggregation_level": aggregation_level,
    }


def _prompt_analysis(llm_input: str, supabase_response: Any) -> dict[str, Any]:
    system_prompt, user_prompt = _split_llm_input(llm_input)
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "data_injected": supabase_response,
        "final_prompt": llm_input,
    }


def _split_llm_input(llm_input: str) -> tuple[str, str]:
    system_prompt = ""
    user_prompt = ""
    if "SYSTEM:\n" in llm_input and "\n\nMESSAGES:\n" in llm_input:
        try:
            system_prompt = llm_input.split("SYSTEM:\n", 1)[1].split("\n\nMESSAGES:\n", 1)[0]
            raw_messages = llm_input.split("\n\nMESSAGES:\n", 1)[1]
            parsed = ast.literal_eval(raw_messages)
            if isinstance(parsed, list):
                user_bits = []
                for msg in parsed:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        user_bits.append(str(msg.get("content", "")))
                user_prompt = "\n".join(user_bits)
            else:
                user_prompt = str(raw_messages)
        except Exception:
            user_prompt = llm_input
    else:
        user_prompt = llm_input
    return system_prompt, user_prompt


def _transformation_detection(llm_output: str, metric: dict[str, Any]) -> dict[str, Any]:
    out = (llm_output or "").lower()
    ops: list[str] = []

    if any(k in out for k in ("average", "mean")):
        ops.append("averaging")
    if any(k in out for k in ("rank", "top", "highest", "lowest", "best", "worst")):
        ops.append("ranking")
    if any(k in out for k in ("sort", "ordered", "descending", "ascending")):
        ops.append("sorting")
    if any(k in out for k in ("compare", "comparison", "versus", "vs", "gap", "difference")):
        ops.append("comparison")
    if any(k in out for k in ("maximum", "max", "highest")):
        ops.append("max detection")
    if any(k in out for k in ("minimum", "min", "lowest")):
        ops.append("min detection")
    if any(k in out for k in ("trend", "over time", "month", "quarter", "year")):
        ops.append("trend detection")
    if any(k in out for k in ("group", "by department", "by category", "by team")):
        ops.append("grouping")

    if metric.get("ranking"):
        if "ranking" not in ops:
            ops.append("ranking")
        if "sorting" not in ops:
            ops.append("sorting")

    return {"operations_detected": ops}


def _metric_reconstruction(rows: list[dict[str, Any]], structural: dict[str, Any]) -> dict[str, Any]:
    numeric_fields = structural.get("numeric_fields", [])
    categorical_fields = structural.get("categorical_fields", [])
    if not rows or not numeric_fields:
        return {}

    metric_col = _pick_metric_column(numeric_fields)
    values = [derivation_trace._to_float(r.get(metric_col)) for r in rows]
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return {}

    out: dict[str, Any] = {
        f"mean_{metric_col}": round(mean(values), 4),
        f"max_{metric_col}": round(max(values), 4),
        f"min_{metric_col}": round(min(values), 4),
        f"variance_{metric_col}": round(pvariance(values), 6) if len(values) > 1 else 0.0,
    }

    cat_col = categorical_fields[0] if categorical_fields else None
    if cat_col:
        ranked: list[dict[str, Any]] = []
        for r in rows:
            c = r.get(cat_col)
            v = derivation_trace._to_float(r.get(metric_col))
            if c is None or v is None:
                continue
            ranked.append({cat_col: c, metric_col: round(v, 4)})
        ranked.sort(key=lambda x: x[metric_col], reverse=True)
        out["ranking"] = ranked[:20]
        out["ranking_basis"] = {"category_field": cat_col, "metric_field": metric_col}

    return out


def _justification_mapping(llm_output: str, metric: dict[str, Any]) -> dict[str, Any]:
    mappings: list[dict[str, Any]] = []
    if not llm_output:
        return {"mappings": mappings}

    sentences = [s.strip() for s in re.split(r"[.!?]\s*", llm_output) if s.strip()]
    ranking = metric.get("ranking", [])
    top = ranking[0] if ranking else None
    bottom = ranking[-1] if ranking else None

    for sentence in sentences[:25]:
        lower = sentence.lower()
        support: dict[str, Any] | None = None
        if top and any(k in lower for k in ("highest", "top", "best", "max")):
            support = top
        elif bottom and any(k in lower for k in ("lowest", "worst", "min")):
            support = bottom
        else:
            for row in ranking[:10]:
                row_text = " ".join(str(v).lower() for v in row.values())
                if any(tok in lower for tok in row_text.split()):
                    support = row
                    break

        if support is not None:
            mappings.append({"statement": sentence, "supported_by": support})

    return {"mappings": mappings}


def _reasoning_trace(
    structural: dict[str, Any],
    semantic: dict[str, Any],
    transform: dict[str, Any],
    metric: dict[str, Any],
) -> dict[str, Any]:
    steps: list[str] = []
    if structural.get("numeric_fields"):
        steps.append("LLM detected numeric fields in Supabase response.")
    if structural.get("categorical_fields"):
        steps.append("LLM detected categorical grouping fields.")
    if semantic.get("metric_type") not in {"unknown", ""}:
        steps.append(f"LLM interpreted primary metric type as {semantic.get('metric_type')}.")
    if semantic.get("aggregation_level") not in {"record", "unknown", ""}:
        steps.append(
            f"LLM inferred aggregation level around {semantic.get('aggregation_level')}."
        )
    for op in transform.get("operations_detected", []):
        steps.append(f"LLM likely applied {op}.")
    if metric.get("ranking"):
        steps.append("Independent metric reconstruction produced ranked results for validation.")

    return {"steps": steps}


def _collect_columns(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                ordered.append(k)
    return ordered


def _is_numeric_series(values: list[Any]) -> bool:
    parsed = [derivation_trace._to_float(v) for v in values]
    good = [v for v in parsed if v is not None and not math.isnan(v)]
    return len(good) >= max(1, int(0.7 * len(values)))


def _pick_metric_column(numeric_fields: list[str]) -> str:
    preferred = ("score", "rating", "count", "percentage", "percent", "value")
    for key in preferred:
        for col in numeric_fields:
            if key in col.lower():
                return col
    return numeric_fields[0]
