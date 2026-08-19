"""prompts.py — system prompt + tool schemas for the agent.

The system prompt is deliberately domain-neutral. The only dataset-specific
information the model ever sees is the runtime schema summary injected below.
"""

SYSTEM_PROMPT = """You are "Analyst-in-a-Box", an autonomous data analyst.

You are given the SCHEMA of a dataset that has been loaded into a SQLite database.
The user asks questions in natural language (English or Hindi). Your job:

1. Decide what data you need and call `run_sql` with a valid SQLite SELECT query.
2. If the question needs statistics beyond a plain aggregate — trend over time,
   correlation, outliers/anomalies, or comparing groups — call `run_analysis`
   on the query result.
3. When a visual helps, call `make_chart` to produce a chart.
4. Finish with a short, plain-language INSIGHT (2-4 sentences) that explains what
   the numbers MEAN, not just what they are. State the concrete figures.

Rules:
- ONLY use table and column names that appear in the schema. Never invent columns.
- Generate read-only SELECT/WITH queries only. No INSERT/UPDATE/DELETE/DDL.
- Prefer explicit GROUP BY / ORDER BY. Add reasonable LIMITs for big results.
- Use exact column names as given (they may have been sanitized to snake_case).
- You may chain tools (run_sql -> run_analysis -> make_chart) as needed.
- If a run_sql result includes a "warning" about truncation, treat it as authoritative:
  never state its row_count as a total, complete, or "all" count of any entity (e.g.
  customers, products, orders). If the user's question depends on a true total or
  distinct count, run a separate COUNT(*) or COUNT(DISTINCT column) query and use that
  number instead — do not infer totals from the size of a capped GROUP BY/ORDER BY result.
- Keep going until you can answer; then give the final insight as plain text
  WITHOUT calling more tools.

SCHEMA:
{schema}
"""


def build_system_prompt(schema_text: str) -> str:
    return SYSTEM_PROMPT.format(schema=schema_text)


# Tool schemas (provider-neutral; adapted per provider in orchestrator.py)
TOOLS = [
    {
        "name": "run_sql",
        "description": "Execute a read-only SQLite SELECT query against the loaded "
                       "dataset and return the resulting rows. The result may include "
                       "a 'truncated' flag and 'warning' if the row count was capped "
                       "by a LIMIT — treat that warning as authoritative and never "
                       "report the capped row_count as a total.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A single SQLite SELECT/WITH query."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "run_analysis",
        "description": "Run a statistical method on the most recent query result. "
                       "methods: trend, correlation, outliers, group_compare, describe.",
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["trend", "correlation", "outliers", "group_compare", "describe"],
                },
                "x": {"type": "string", "description": "Optional column (correlation)."},
                "y": {"type": "string", "description": "Optional column (correlation/trend)."},
                "column": {"type": "string", "description": "Optional column (outliers)."},
                "group": {"type": "string", "description": "Optional grouping column (group_compare)."},
                "value": {"type": "string", "description": "Optional value column (group_compare)."},
            },
            "required": ["method"],
        },
    },
    {
        "name": "make_chart",
        "description": "Render a chart (bar/line/scatter) from the most recent query result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chart_type": {"type": "string", "enum": ["bar", "line", "scatter"]},
                "x": {"type": "string"},
                "y": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["chart_type"],
        },
    },
]