"""
Query Agent — NL-to-SQL
- Loads schema graph (NetworkX) for join path resolution
- Sends NL + schema context + join paths to Claude Haiku
- LLM returns SQL + chart config as JSON
- Executes SQL across operations.db + quality.db via SQLite ATTACH
- Returns: DataFrame results + chart config
"""

import os
import json
import re
import sqlite3
import pandas as pd
from dotenv import load_dotenv
import anthropic

from schema_graph import load_artifacts, build_graph, describe_graph, get_all_join_paths
from semantic_inferrer import load_ontology, build_semantic_graph, get_semantic_context

load_dotenv()

OPERATIONS_DB = "operations.db"
QUALITY_DB    = "quality.db"

# ── Bootstrap graph once at import ────────────────────────────────────────────
_relationships, _schema_context = load_artifacts()
GRAPH          = build_graph(_relationships, _schema_context)
ALL_JOIN_PATHS = get_all_join_paths(GRAPH)

_ontology      = load_ontology("ontology_draft.json")
GRAPH_DESCRIPTION = get_semantic_context(_ontology)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ── Build join paths block for prompt ─────────────────────────────────────────
def _format_join_paths():
    lines = ["AVAILABLE JOIN PATHS (pre-computed from schema graph):"]
    for pair, info in ALL_JOIN_PATHS.items():
        lines.append(f"\n  {pair} ({info['hops']} hop):")
        for clause in info["join_clauses"]:
            lines.append(f"    {clause}")
    return "\n".join(lines)


# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert SQL agent for an industrial manufacturing database.
You convert natural language questions into SQLite SQL queries.

IMPORTANT RULES:
1. The data lives in TWO SQLite databases connected via ATTACH:
   - Main DB (operations): tables are machines, maintenance_logs, parts_inventory
     → NO prefix needed. Write: machines, maintenance_logs, parts_inventory
   - Attached DB (quality): tables need the quality. prefix
     → Write: quality.production_batches, quality.defect_reports

2. CORRECT examples:
   SELECT machines.machine_type FROM machines                          ✓
   SELECT * FROM quality.defect_reports                                ✓
   JOIN maintenance_logs ON machines.machine_id = maintenance_logs.machine_id  ✓

3. WRONG examples:
   SELECT * FROM operations.machines     ✗  (never prefix operations tables)
   SELECT * FROM defect_reports          ✗  (quality tables need prefix)

4. Column name is downtime_hrs not downtime_hours, repair_cost_usd not repair_cost.
   Always use exact column names from the schema context provided.

5. Keep queries efficient — use LIMIT 500 unless aggregating.

6. Return ONLY valid JSON, no markdown:
{
  "sql": "<your SQL query>",
  "chart": {
    "type": "bar" | "line" | "scatter" | "none",
    "x": "<column name>",
    "y": "<column name>",
    "title": "<chart title>"
  },
  "explanation": "<one sentence>"
}
"""


def _build_user_prompt(nl_query):
    return f"""
{GRAPH_DESCRIPTION}

{_format_join_paths()}

USER QUESTION:
{nl_query}

Return only the JSON object.
""".strip()


# ── LLM call ──────────────────────────────────────────────────────────────────
def _call_llm(nl_query):
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(nl_query)}],
    )
    raw = response.content[0].text.strip()

    # Strip markdown fences if model wraps anyway
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON block from response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"LLM returned non-JSON:\n{raw}")


# ── SQL execution ──────────────────────────────────────────────────────────────
def _execute_sql(sql):
    """
    Runs SQL against operations.db with quality.db ATTACHed as 'quality'.
    Returns a DataFrame.
    """
    conn = sqlite3.connect(OPERATIONS_DB)
    try:
        conn.execute(f"ATTACH DATABASE '{QUALITY_DB}' AS quality")
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return df


# ── Public interface ───────────────────────────────────────────────────────────
def run_query(nl_query: str) -> dict:
    """
    Main entry point. Takes a natural language question, returns:
    {
        "sql":         str,
        "explanation": str,
        "chart":       dict,
        "data":        pd.DataFrame,
        "error":       str | None,
    }
    """
    result = {
        "sql":         None,
        "explanation": None,
        "chart":       {"type": "none"},
        "data":        pd.DataFrame(),
        "error":       None,
    }

    try:
        llm_output = _call_llm(nl_query)
        result["sql"]         = llm_output.get("sql", "")
        result["explanation"] = llm_output.get("explanation", "")
        result["chart"]       = llm_output.get("chart", {"type": "none"})
    except Exception as e:
        result["error"] = f"LLM error: {e}"
        return result

    try:
        result["data"] = _execute_sql(result["sql"])
    except Exception as e:
        result["error"] = f"SQL execution error: {e}\n\nGenerated SQL:\n{result['sql']}"

    return result


# ── CLI test harness ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    TEST_QUERIES = [
        "Which machine types have the highest average downtime hours?",
        "Show total defects by severity across all production batches",
        "Which plants have the most critical maintenance issues?",
        "What is the trend of defect rate by product line?",
        "Which machines have both high repair costs and high defect rates?",
    ]

    for q in TEST_QUERIES:
        print("\n" + "═" * 60)
        print(f"Q: {q}")
        print("═" * 60)

        out = run_query(q)

        if out["error"]:
            print(f"  ✗ ERROR: {out['error']}")
            continue

        print(f"  Explanation : {out['explanation']}")
        print(f"  Chart       : {out['chart']['type']} — x={out['chart'].get('x')} y={out['chart'].get('y')}")
        print(f"  SQL         :\n    {out['sql']}")
        print(f"  Rows        : {len(out['data'])}")
        if not out["data"].empty:
            print(out["data"].head(5).to_string(index=False))