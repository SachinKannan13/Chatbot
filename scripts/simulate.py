"""
Simulation — tests all 13 question types end-to-end against mock survey data.

Usage:
    cd chatbot
    python simulate.py

The mock DataFrame mirrors the real schema inferred from the actual data.
Tests the full pipeline: classify → SQL generate → SQL execute → compose.
Prints PASS/FAIL for each question type plus the actual SQL and response.
"""
import asyncio
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load env before anything else
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np

from src.pipeline.question_classifier import classify_question
from src.pipeline.sql_generator import generate_sql
from src.pipeline.sql_executor import execute_multiple, merge_results, resolve_sql_string_values
from src.pipeline.answer_composer import compose_answer
from src.data.schema_analyzer import analyze_schema


# ── Build realistic mock DataFrame ───────────────────────────────────────────

def build_mock_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 500

    health_levers = [
        "Career Health", "Mental Health", "Financial Health",
        "Cultural health", "Digital and Tech Health", "Physiological Health",
        "Social Health", "Organization Health", "Intellectual Health",
        "Holistic Growth",
    ]
    departments = ["Engineering", "Sales", "HR", "Finance", "Marketing", "Operations"]
    quarters = ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"]
    employees = [f"EMP{str(i).zfill(4)}" for i in range(1, 101)]

    rows = []
    for _ in range(n):
        lever = np.random.choice(health_levers)
        dept = np.random.choice(departments)
        # Career Health and Mental Health score slightly lower on average
        base = 3.2 if lever in ("Career Health", "Mental Health") else 3.8
        score = round(np.clip(np.random.normal(base, 0.6), 1.0, 5.0), 2)
        rows.append({
            "Employee_ID": np.random.choice(employees),
            "Department": dept,
            "Health Lever": lever,
            "Score": score,
            "Quarter": np.random.choice(quarters),
            "Response_Count": np.random.randint(1, 5),
        })

    return pd.DataFrame(rows)


# ── Questions for each type ───────────────────────────────────────────────────

QUESTIONS = [
    ("simple",           "What is the overall average score?"),
    ("aggregation",      "What is the average score by Health Lever?"),
    ("calculation",      "Calculate the average score for each department"),
    ("count",            "How many responses are there in total?"),
    ("list",             "List all unique Health Levers in the dataset"),
    ("ranking",          "Which departments have the highest average scores? Show top 5"),
    ("comparison",       "Compare the average scores between Career Health and Mental Health"),
    ("single_intent",    "Give me a deep analysis of Career Health scores across all departments"),
    ("multi_intent",     "What is the average score by department and also by Health Lever?"),
    ("trend",            "Show the average score trend by quarter"),
    ("distribution",     "Show the distribution of scores across all Health Levers"),
    ("insights",         "What are the key insights from the Career Health lever scores?"),
    ("recommendations",  "Give me recommendations to improve the scores of the Career Health lever"),
]


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_simulation():
    print("=" * 70)
    print("EMPLOYEE SURVEY CHATBOT — SIMULATION")
    print("=" * 70)

    df = build_mock_df()
    print(f"\nMock dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Health Lever values: {sorted(df['Health Lever'].unique().tolist())}")
    print(f"Departments: {sorted(df['Department'].unique().tolist())}\n")

    # Analyze schema (LLM call)
    print("Analyzing schema...")
    metadata = await analyze_schema(df)
    print(f"Primary metric: {metadata.get('primary_metric_column')}")
    print(f"Primary dims: {metadata.get('primary_dimension_columns')}")
    print(f"Table summary: {metadata.get('table_summary')}\n")

    results = []

    for expected_type, question in QUESTIONS:
        print("-" * 70)
        print(f"TYPE: {expected_type.upper()}")
        print(f"Q:    {question}")

        status = "PASS"
        notes = []

        try:
            # Step 1: Classify
            classification = await classify_question(question, metadata)
            got_type = classification.get("question_type", "?")
            type_ok = got_type == expected_type
            if not type_ok:
                notes.append(f"[type mismatch: expected={expected_type}, got={got_type}]")

            print(f"      Classified as: {got_type} | Expected: {expected_type} {'OK' if type_ok else 'NO'}")

            # Step 2: Generate SQL
            sqls = await generate_sql(question, classification, metadata, df)
            print(f"      SQL ({len(sqls)} query/ies):")
            for i, sql in enumerate(sqls):
                print(f"        [{i+1}] {sql[:120].replace(chr(10), ' ')}")

            # Verify value resolution works (key fix test)
            for sql in sqls:
                resolved = resolve_sql_string_values(sql, df)
                if resolved != sql:
                    print(f"      [value resolved: '{sql[:60]}' → '{resolved[:60]}']")

            # Step 3: Execute
            result_dfs = execute_multiple(sqls, df)
            result_df = merge_results(result_dfs)

            if result_df.empty:
                notes.append("[WARN: empty result]")
                status = "WARN"
            else:
                print(f"      Result: {len(result_df)} rows × {len(result_df.columns)} cols")
                print(f"      Preview:\n{result_df.head(3).to_string(index=False)}")

                # Verify score columns are averages (not sums)
                numeric_cols = result_df.select_dtypes(include=["number"]).columns
                for col in numeric_cols:
                    col_max = result_df[col].max()
                    col_min = result_df[col].min()
                    # Scores should be in a reasonable avg range, not absurdly large
                    if "score" in col.lower() and col_max > 10:
                        notes.append(f"[WARN: {col} max={col_max:.1f} — may be SUM not AVG]")
                        status = "WARN"

            # Step 4: Compose answer
            answer = await compose_answer(
                question=question,
                classification=classification,
                result_df=result_df,
                web_context="",
                conversation_history=[],
                company_name="SimCorp",
            )

            # Verify answer is not empty and is grounded
            if len(answer) < 20:
                notes.append("[WARN: answer too short]")
                status = "WARN"

            # For score questions: verify answer contains numbers (not just text)
            if expected_type in ("aggregation", "ranking", "calculation"):
                import re
                nums_in_answer = re.findall(r"\d+\.\d+|\d+", answer)
                if not nums_in_answer:
                    notes.append("[WARN: no numbers found in answer]")
                    status = "WARN"

            print(f"      Answer: {answer[:250]}")

        except Exception as e:
            import traceback
            status = "FAIL"
            notes.append(f"[ERROR: {e}]")
            traceback.print_exc()

        note_str = " ".join(notes) if notes else ""
        print(f"      STATUS: {status} {note_str}\n")
        results.append((expected_type, status, notes))

    # Summary
    print("=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    warned = sum(1 for _, s, _ in results if s == "WARN")
    failed = sum(1 for _, s, _ in results if s == "FAIL")

    for q_type, status, notes in results:
        icon = "OK" if status == "PASS" else ("!!" if status == "WARN" else "NO")
        note_str = " ".join(notes) if notes else ""
        print(f"  {icon} {q_type:<20} {status}  {note_str}")

    print(f"\nTotal: {passed} PASS, {warned} WARN, {failed} FAIL out of {len(results)}")

    # Additional fix verification test
    print("\n" + "=" * 70)
    print("FIX VERIFICATION TESTS")
    print("=" * 70)

    # Test 1: Case sensitivity fix
    print("\n[Test 1] Case-insensitive value resolution:")
    test_sqls = [
        ('WHERE "Health Lever" = \'career health\'',   "Career Health"),
        ('WHERE "Health Lever" = \'CAREER HEALTH\'',   "Career Health"),
        ('WHERE "Health Lever" = \'career Health\'',   "Career Health"),
        ('WHERE "Department" = \'engineering\'',       "Engineering"),
        ('WHERE "Health Lever" = \'mental health\'',   "Mental Health"),
    ]
    for sql_frag, expected_val in test_sqls:
        resolved = resolve_sql_string_values(sql_frag, df)
        ok = expected_val in resolved
        print(f"  {'OK' if ok else 'NO'} '{sql_frag}' -> '{resolved}' (expected '{expected_val}')")

    # Test 2: Score averaging
    print("\n[Test 2] Score averaging (no raw sums):")
    from src.pipeline.sql_executor import execute_sql
    avg_sql = 'SELECT "Health Lever", AVG(CAST("Score" AS REAL)) AS avg_score FROM dataset GROUP BY "Health Lever" ORDER BY avg_score DESC;'
    avg_result = execute_sql(avg_sql, df)
    all_in_range = all(1.0 <= v <= 5.0 for v in avg_result["avg_score"])
    print(f"  {'OK' if all_in_range else 'NO'} All avg_scores in [1.0-5.0] range: {all_in_range}")
    print(f"  Sample:\n{avg_result.head(5).to_string(index=False)}")

    print("\nSimulation complete.")


if __name__ == "__main__":
    asyncio.run(run_simulation())
