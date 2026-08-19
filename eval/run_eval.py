"""run_eval.py — run the agent against test_questions.json and report accuracy.

Proves the agent generalizes: the SAME agent code runs against 3 unrelated
datasets (sales / sports / education) with no per-dataset changes.

Ground truth for value questions is computed by executing an independent
`expected_sql` directly against the DB — so we compare the agent to reality,
not to a hardcoded answer.

Usage:
    python -m eval.run_eval                # run all
    python -m eval.run_eval --provider openai
    python -m eval.run_eval --limit 3      # smoke test

Requires the relevant API key in the environment.
"""

from __future__ import annotations

import os
import re
import sys
import json
import argparse
import tempfile

# make project root importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# load a local .env (gitignored) if present, so ANTHROPIC_API_KEY etc. are picked up
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except Exception:
    pass

from agent import schema_reader
from agent.tools import run_sql
from agent.orchestrator import answer_question

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SAMPLES = os.path.join(ROOT, "data", "sample_datasets")


def _build_db(dataset: str) -> tuple[str, str]:
    """Load a sample CSV into a temp SQLite DB, return (db_path, schema_text)."""
    db_path = os.path.join(tempfile.mkdtemp(prefix=f"eval_{dataset}_"), "d.db")
    csv = os.path.join(SAMPLES, f"{dataset}.csv")
    schema_reader.load_flat_file_to_sqlite(csv, db_path, table_name=dataset, file_type="csv")
    schema = schema_reader.introspect(db_path)
    return db_path, schema.to_prompt_text()


def _nums(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text or "")]


def _value_matches(answer: str, expected, tolerance: float) -> bool:
    if expected is None:
        return False
    exp_str = str(expected).strip()
    # non-numeric expected (e.g. a region/team name): substring match, case-insensitive
    try:
        exp_val = float(str(exp_str).replace(",", ""))
        is_num = True
    except ValueError:
        is_num = False

    if not is_num:
        return exp_str.lower() in (answer or "").lower()

    for n in _nums(answer):
        if exp_val == 0:
            if abs(n) < 1e-9:
                return True
        elif abs(n - exp_val) <= abs(exp_val) * max(tolerance, 1e-6):
            return True
        # also accept exact integer match ignoring rounding
        if abs(round(n) - round(exp_val)) < 1:
            return True
    return False


def evaluate(tests: list[dict], provider: str | None) -> list[dict]:
    dbs: dict[str, tuple[str, str]] = {}
    results = []

    for i, t in enumerate(tests, 1):
        ds = t["dataset"]
        if ds not in dbs:
            dbs[ds] = _build_db(ds)
        db_path, schema_text = dbs[ds]

        res = answer_question(t["question"], db_path, schema_text, provider=provider)

        passed = False
        detail = ""
        check = t["check"]

        if check == "value_in_answer":
            gt = run_sql(t["expected_sql"], db_path)
            expected = gt["rows"][0][gt["columns"][0]] if gt["rows"] else None
            passed = _value_matches(res.answer, expected, t.get("tolerance", 0.02))
            detail = f"expected~={expected}"
        elif check == "chart_made":
            passed = len(res.charts) > 0
            detail = f"charts={len(res.charts)}"
        elif check == "analysis_method":
            methods = [a.get("method") for a in res.analyses]
            passed = t["expected_method"] in methods
            detail = f"methods={methods}, wanted={t['expected_method']}"

        if res.error:
            detail += f" | error={res.error}"

        results.append({
            "n": i, "dataset": ds, "question": t["question"],
            "check": check, "passed": passed, "detail": detail,
            "sql": res.sql_queries,
        })
        status = "PASS" if passed else "FAIL"
        print(f"[{i:>2}/{len(tests)}] {status}  ({ds}) {t['question']}")
        print(f"        {detail}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "anthropic"))
    ap.add_argument("--limit", type=int, default=0, help="only run first N tests")
    ap.add_argument("--out", default=os.path.join(HERE, "eval_results.json"))
    args = ap.parse_args()

    with open(os.path.join(HERE, "test_questions.json"), encoding="utf-8") as f:
        tests = json.load(f)
    if args.limit:
        tests = tests[: args.limit]

    results = evaluate(tests, args.provider)
    n_pass = sum(r["passed"] for r in results)
    rate = n_pass / len(results) if results else 0.0

    by_ds: dict[str, list[bool]] = {}
    for r in results:
        by_ds.setdefault(r["dataset"], []).append(r["passed"])

    print("\n" + "=" * 60)
    print(f"OVERALL: {n_pass}/{len(results)} passed ({rate:.0%})")
    for ds, vals in by_ds.items():
        print(f"  {ds:<10} {sum(vals)}/{len(vals)} ({sum(vals)/len(vals):.0%})")
    print("=" * 60)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"rate": rate, "passed": n_pass, "total": len(results),
                   "results": results}, f, indent=2, default=str)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
