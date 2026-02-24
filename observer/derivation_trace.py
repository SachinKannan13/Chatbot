from __future__ import annotations

import math
import re
from statistics import mean, median, multimode, pstdev, pvariance
from typing import Any


def analyze_derivations(
    supabase_response: Any,
    llm_input: str,
    llm_output: str,
    structural_analysis: dict[str, Any] | None = None,
    semantic_analysis: dict[str, Any] | None = None,
    metric_reconstruction: dict[str, Any] | None = None,
    transformation_detection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Passive mathematical/logical derivation analyzer.
    Returns a structured derivation trace object and never mutates runtime behavior.
    """
    try:
        rows = _extract_rows(supabase_response)
        structural = structural_analysis or {}
        numeric_fields = list(structural.get("numeric_fields", [])) or _infer_numeric_fields(rows)
        values_by_field = {f: _extract_numeric_values(rows, f) for f in numeric_fields}
        metric_field = _pick_metric_field(values_by_field)
        values = values_by_field.get(metric_field, []) if metric_field else []

        text = f"{llm_input}\n{llm_output}".lower()
        ops = _detect_mathematical_operations(text, values)
        formulas, reconstructions = _reconstruct_formulas(ops, values, rows, metric_field)
        logical = _detect_logical_derivations(text, values)
        inference_rules = _detect_inference_rules(text, rows, metric_field, values)
        confidence = _build_confidence_scores(ops, text, values)

        return {
            "mathematical_operations": ops,
            "formulas_detected": formulas,
            "formula_reconstructions": reconstructions,
            "logical_derivations": logical,
            "inference_rules": inference_rules,
            "derivation_confidence": confidence,
        }
    except Exception as e:
        return {
            "error": f"derivation_trace_failed: {e}",
            "mathematical_operations": [],
            "formulas_detected": [],
            "formula_reconstructions": [],
            "logical_derivations": [],
            "inference_rules": [],
            "derivation_confidence": [],
        }


def _extract_rows(supabase_response: Any) -> list[dict[str, Any]]:
    raw = supabase_response
    if isinstance(raw, dict) and "raw_response" in raw:
        raw = raw.get("raw_response")
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        raw = raw["data"]
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    return []


def _infer_numeric_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    if not rows:
        return fields
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    for key in keys:
        vals = [_to_float(r.get(key)) for r in rows]
        valid = [v for v in vals if v is not None and not math.isnan(v)]
        if valid and len(valid) >= max(1, int(0.6 * len(rows))):
            fields.append(key)
    return fields


def _extract_numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        v = _to_float(row.get(field))
        if v is not None and not math.isnan(v):
            values.append(v)
    return values


def _pick_metric_field(values_by_field: dict[str, list[float]]) -> str | None:
    if not values_by_field:
        return None
    preferred = ("score", "rating", "percent", "percentage", "count", "value", "avg", "mean")
    for kw in preferred:
        for field in values_by_field.keys():
            if kw in field.lower():
                return field
    return next(iter(values_by_field.keys()), None)


def _detect_mathematical_operations(text: str, values: list[float]) -> list[str]:
    ops: list[str] = []
    map_ops = {
        "mean": ("mean", "average", "avg"),
        "median": ("median",),
        "mode": ("mode", "most common"),
        "sum": ("sum", "total"),
        "difference": ("difference", "gap", "minus"),
        "percentage calculation": ("percentage", "percent", "%"),
        "ratio calculation": ("ratio", "per ", "x:"),
        "variance": ("variance",),
        "standard deviation": ("standard deviation", "std dev", "std"),
        "ranking formula": ("rank", "ranking", "top", "bottom", "highest", "lowest"),
        "normalization": ("normalize", "normalized", "scaled"),
        "min/max detection": ("min", "max", "highest", "lowest"),
    }
    for name, keys in map_ops.items():
        if any(k in text for k in keys):
            ops.append(name)

    if values:
        if "min/max detection" not in ops:
            ops.append("min/max detection")
        if "mean" not in ops and len(values) > 1:
            ops.append("mean")
    return ops


def _reconstruct_formulas(
    operations: list[str],
    values: list[float],
    rows: list[dict[str, Any]],
    metric_field: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    formulas_detected: list[dict[str, Any]] = []
    recon: list[dict[str, Any]] = []
    if not values:
        return formulas_detected, recon

    def add_formula(name: str, expr: str, result: Any, extra: dict[str, Any] | None = None) -> None:
        base = {
            "formula_name": name,
            "formula_expression": expr,
            "result": result,
        }
        if extra:
            base.update(extra)
        formulas_detected.append(base)
        recon.append(base.copy())

    if "mean" in operations:
        add_formula(
            "mean",
            "mean = sum(values) / count(values)",
            round(mean(values), 6),
            {"values_used": _clip(values)},
        )
    if "median" in operations:
        add_formula("median", "median(values)", round(median(values), 6), {"values_used": _clip(values)})
    if "mode" in operations:
        modes = multimode(values)
        add_formula("mode", "mode(values)", _clip(modes, 10), {"values_used": _clip(values)})
    if "sum" in operations:
        add_formula("sum", "sum(values)", round(sum(values), 6), {"values_used": _clip(values)})
    if "variance" in operations:
        add_formula(
            "variance",
            "variance = mean((x - mean(values))^2)",
            round(pvariance(values), 6) if len(values) > 1 else 0.0,
            {"values_used": _clip(values)},
        )
    if "standard deviation" in operations:
        add_formula(
            "standard_deviation",
            "std = sqrt(variance(values))",
            round(pstdev(values), 6) if len(values) > 1 else 0.0,
            {"values_used": _clip(values)},
        )
    if "min/max detection" in operations:
        add_formula("max", "max(values)", round(max(values), 6), {"values_used": _clip(values)})
        add_formula("min", "min(values)", round(min(values), 6), {"values_used": _clip(values)})

    if "percentage calculation" in operations:
        total = sum(values)
        if total:
            pct = (max(values) / total) * 100.0
            add_formula(
                "percentage",
                "percentage = (value / total) * 100",
                round(pct, 6),
                {"value_used": round(max(values), 6), "total": round(total, 6)},
            )

    if "ratio calculation" in operations and len(values) >= 2 and values[1] != 0:
        ratio = values[0] / values[1]
        add_formula(
            "ratio",
            "ratio = value_a / value_b",
            round(ratio, 6),
            {"value_a": round(values[0], 6), "value_b": round(values[1], 6)},
        )

    if "ranking formula" in operations and metric_field:
        ranked = _ranking_from_rows(rows, metric_field)
        add_formula(
            "ranking",
            "sorted(values, descending=True)",
            "ranked_output",
            {
                "input_values": _clip(values),
                "ranked_output": ranked[:20],
            },
        )

    if "difference" in operations and len(values) >= 2:
        diff = max(values) - min(values)
        add_formula(
            "difference",
            "difference = max(values) - min(values)",
            round(diff, 6),
            {"max_value": round(max(values), 6), "min_value": round(min(values), 6)},
        )

    if "normalization" in operations:
        vmin = min(values)
        vmax = max(values)
        norm = []
        if vmax != vmin:
            norm = [round((v - vmin) / (vmax - vmin), 6) for v in values[:20]]
        add_formula(
            "normalization",
            "normalized = (x - min(values)) / (max(values) - min(values))",
            "vector",
            {"normalized_values": norm},
        )

    return formulas_detected, recon


def _ranking_from_rows(rows: list[dict[str, Any]], metric_field: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not rows:
        return out
    category_field = None
    for key in rows[0].keys():
        if key != metric_field:
            category_field = key
            break
    if category_field is None:
        category_field = "item"

    for idx, row in enumerate(rows):
        score = _to_float(row.get(metric_field))
        if score is None:
            continue
        item = row.get(category_field, f"item_{idx+1}")
        out.append({str(category_field): item, str(metric_field): round(score, 6)})
    out.sort(key=lambda x: x[metric_field], reverse=True)
    return out


def _detect_logical_derivations(text: str, values: list[float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not values:
        return out
    avg = mean(values)

    if any(k in text for k in ("highest", "top", "max", "best")):
        out.append(
            {
                "logic_type": "max_detection",
                "formula_expression": "max(values)",
                "result": round(max(values), 6),
            }
        )
    if any(k in text for k in ("lowest", "bottom", "min", "worst")):
        out.append(
            {
                "logic_type": "min_detection",
                "formula_expression": "min(values)",
                "result": round(min(values), 6),
            }
        )
    if any(k in text for k in ("above average", "above mean", "higher than average")):
        out.append(
            {
                "logic_type": "above_average_detection",
                "formula_expression": "value > mean(values)",
                "result": {"mean": round(avg, 6)},
            }
        )
    if any(k in text for k in ("below average", "below mean", "lower than average")):
        out.append(
            {
                "logic_type": "below_average_detection",
                "formula_expression": "value < mean(values)",
                "result": {"mean": round(avg, 6)},
            }
        )
    if any(k in text for k in ("greater than", "less than", "above", "below", "threshold", ">=", "<=", ">", "<")):
        out.append(
            {
                "logic_type": "threshold_comparison",
                "formula_expression": "value comparator threshold",
                "result": "detected",
            }
        )
    return out


def _detect_inference_rules(
    text: str,
    rows: list[dict[str, Any]],
    metric_field: str | None,
    values: list[float],
) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    if not values:
        return rules

    rule_patterns = [
        ("strong", r"(?:score|rating)\s*>\s*80"),
        ("weak", r"(?:score|rating)\s*<\s*60"),
        ("high", r"(?:score|rating)\s*>\s*mean"),
        ("low", r"(?:score|rating)\s*<\s*mean"),
    ]
    for label, pattern in rule_patterns:
        if re.search(pattern, text):
            rules.append(
                {
                    "inference_rule": re.search(pattern, text).group(0),  # type: ignore[union-attr]
                    "applied_to": round(max(values), 6),
                    "conclusion": f"Detected {label} classification inference.",
                }
            )

    if any(k in text for k in ("strong", "high performing", "excellent")):
        top_val = round(max(values), 6)
        top_label = _best_label(rows, metric_field, top_val)
        rules.append(
            {
                "inference_rule": "score > 80 = strong performance",
                "applied_to": top_val,
                "conclusion": f"{top_label} is strong" if top_label else "Top item inferred as strong.",
            }
        )

    if any(k in text for k in ("weak", "needs improvement", "poor")):
        low_val = round(min(values), 6)
        low_label = _best_label(rows, metric_field, low_val)
        rules.append(
            {
                "inference_rule": "low score = needs improvement",
                "applied_to": low_val,
                "conclusion": f"{low_label} needs improvement" if low_label else "Lowest item inferred weak.",
            }
        )
    return rules


def _build_confidence_scores(
    operations: list[str],
    text: str,
    values: list[float],
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for op in operations:
        confidence = 0.55
        if op in {"mean", "min/max detection", "ranking formula"} and values:
            confidence = 0.9
        if op == "mean" and any(k in text for k in ("average", "mean", "avg")):
            confidence = 0.98
        if op == "ranking formula" and any(k in text for k in ("highest", "lowest", "top", "bottom", "rank")):
            confidence = 0.97
        if op == "percentage calculation" and ("%" in text or "percent" in text):
            confidence = 0.95
        if op in {"variance", "standard deviation", "mode", "median"} and op.split()[0] in text:
            confidence = 0.92
        scores.append({"formula": op, "confidence": round(confidence, 2)})
    return scores


def _best_label(rows: list[dict[str, Any]], metric_field: str | None, target: float) -> str:
    if not rows or not metric_field:
        return ""
    for row in rows:
        value = _to_float(row.get(metric_field))
        if value is None:
            continue
        if abs(value - target) < 1e-6:
            for k, v in row.items():
                if k != metric_field:
                    return str(v)
    return ""


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").rstrip("%")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _clip(values: list[Any], limit: int = 20) -> list[Any]:
    return values[:limit]

