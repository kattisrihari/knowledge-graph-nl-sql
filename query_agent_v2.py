"""
query_agent_v2.py
─────────────────
NL-to-SQL agent for Siemens Energy domain.
- Two SQLite DBs: domain1.db (Power Grid) + domain2.db (Campaigns)
- Semantic context loaded from ontology.json (LLM-inferred)
- Cross-DB joins via SQLite ATTACH
- Returns: DataFrame + chart config

DB prefix rules (SQLite ATTACH):
  domain1 tables → NO prefix  : ASSET_VW, OUTAGE__C_VW, CASE_VW, SERVICE_WINDOW_C_VW_TEST
  domain2 tables → domain2.   : domain2.USER_VW, domain2.PRODUCT2_VW, domain2.CAMPAIGN_VW, domain2.EVENT_VW
"""

import os
import json
import re
import sqlite3
import pandas as pd
from dotenv import load_dotenv
import anthropic

from semantic_inferrer_v2 import get_semantic_context

def load_ontology(path: str = "ontology.json") -> dict:
    with open(path) as f:
        return json.load(f)
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

DOMAIN1_DB = "domain1.db"
DOMAIN2_DB = "domain2.db"

def _build_join_paths(ontology: dict) -> str:
    """Format all edges as JOIN clauses for the LLM prompt."""
    lines = ["AVAILABLE JOIN PATHS (from semantic ontology):"]
    for edge in ontology["edges"]:
        from_t  = edge["from_table"].split(".")[-1]
        to_t    = edge["to_table"].split(".")[-1]
        from_db = edge["from_table"].split(".")[0]
        to_db   = edge["to_table"].split(".")[0]

        # Apply SQLite prefix rules
        from_sql = f"domain2.{from_t}" if from_db == "domain2" else from_t
        to_sql   = f"domain2.{to_t}"   if to_db   == "domain2" else to_t

        lines.append(
            f"  {edge['from_label']} --[{edge['relationship']}]--> {edge['to_label']}"
            f"\n    JOIN {to_sql} ON {from_sql}.{edge['from_col']} = {to_sql}.{edge['to_col']}"
        )
    return "\n".join(lines)


# ── Bootstrap semantic context once at import ──────────────────────
_ontology        = load_ontology("ontology.json")
SEMANTIC_CONTEXT = get_semantic_context(_ontology)
JOIN_PATHS       = _build_join_paths(_ontology)


# ── System prompt ──────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert SQL agent for a Siemens Energy industrial database.
You convert natural language questions into SQLite SQL queries.

CRITICAL PREFIX RULES — SQLite ATTACH:
  domain1 tables (main DB) → NO prefix:
    ASSET_VW, OUTAGE__C_VW, CASE_VW, SERVICE_WINDOW_C_VW_TEST

  domain2 tables (attached) → MUST use domain2. prefix:
    domain2.USER_VW, domain2.PRODUCT2_VW, domain2.CAMPAIGN_VW, domain2.EVENT_VW

CORRECT examples:
  SELECT * FROM ASSET_VW                              ✓
  SELECT * FROM domain2.CAMPAIGN_VW                  ✓
  JOIN domain2.USER_VW ON domain2.CAMPAIGN_VW.OWNER_ID = domain2.USER_VW.USER_ID  ✓

WRONG examples:
  SELECT * FROM domain1.ASSET_VW                     ✗  never prefix domain1 tables
  SELECT * FROM USER_VW                              ✗  domain2 tables need prefix

QUERY RULES:
1. Use only the join paths provided — do not invent joins
2. Use exact enum values from slot context for WHERE clauses
3. For aggregations return all rows; otherwise LIMIT 500
4. Column names are UPPERCASE — use them exactly as shown

Return ONLY valid JSON, no markdown:
{
  "sql": "<SQLite query>",
  "chart": {
    "type": "bar" | "line" | "scatter" | "none",
    "x": "<column>",
    "y": "<column>",
    "title": "<chart title>"
  },
  "explanation": "<one sentence>"
}"""


def _build_user_prompt(nl_query: str) -> str:
    return f"""{SEMANTIC_CONTEXT}

{JOIN_PATHS}

USER QUESTION:
{nl_query}

EXACT COLUMN NAMES — use these exactly, no variations:
  ASSET_VW:                  ASSET_ID, PLANT_LOCATION_ID, SERIAL_NUMBER, EQUIPMENT_CLASS, MODEL, COMMISSIONING_DATE, OPERATIONAL_STATUS, REGION, RATED_CAPACITY_MW, LAST_INSPECTION_DATE
  OUTAGE__C_VW:              OUTAGE_ID, ASSET_ID, TRIP_REASON, SEVERITY, START_TIME, RESTORED_TIME, DOWNTIME_HRS, LOSS_MWH, REPORTED_BY
  CASE_VW:                   CASE_ID, ASSET_ID, OUTAGE_ID, ISSUE_CATEGORY, PRIORITY, STATUS, CREATED_AT, ASSIGNED_TO, RESOLUTION_NOTES, ESTIMATED_COST_EUR
  SERVICE_WINDOW_C_VW_TEST:  WINDOW_ID, ASSET_ID, WINDOW_TYPE, APPROVED_START, APPROVED_END, DURATION_DAYS, SAFETY_SIGN_OFF_BY, APPROVAL_STATUS, REGION
  domain2.USER_VW:           USER_ID, FULL_NAME, EMAIL, SPECIALIZATION, REGION, ROLE, YEARS_EXPERIENCE, PHONE
  domain2.PRODUCT2_VW:       PRODUCT_ID, PART_NUMBER, NAME, PRODUCT_FAMILY, UNIT_COST_EUR, LEAD_TIME_DAYS, STOCK_AVAILABLE, COMPATIBILITY
  domain2.CAMPAIGN_VW:       CAMPAIGN_ID, CAMPAIGN_NAME, OWNER_ID, OBJECTIVE, BUDGET_EUR, STATUS, START_DATE, END_DATE, REGION, APPROVED_BY
  domain2.EVENT_VW:          EVENT_ID, CAMPAIGN_ID, PRODUCT_ID, EVENT_TYPE, PLANNED_DATE, EXECUTION_STATUS, ASSIGNED_ENGINEER, ACTUAL_DATE, COST_EUR, NOTES

Return only the JSON object."""


# ── LLM call ──────────────────────────────────────────────────────
def _call_llm(nl_query: str) -> dict:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(nl_query)}],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$",     "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"LLM returned non-JSON:\n{raw}")


# ── SQL execution ──────────────────────────────────────────────────
def _execute_sql(sql: str) -> pd.DataFrame:
    """
    Runs SQL against domain1.db with domain2.db ATTACHed as 'domain2'.
    """
    conn = sqlite3.connect(DOMAIN1_DB)
    try:
        conn.execute(f"ATTACH DATABASE '{DOMAIN2_DB}' AS domain2")
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return df


# ── Public interface ───────────────────────────────────────────────
def run_query(nl_query: str) -> dict:
    """
    Main entry point. Returns:
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
        llm_output        = _call_llm(nl_query)
        result["sql"]         = llm_output.get("sql", "")
        result["explanation"] = llm_output.get("explanation", "")
        result["chart"]       = llm_output.get("chart", {"type": "none"})
    except Exception as e:
        result["error"] = f"LLM error: {e}"
        return result

    try:
        result["data"] = _execute_sql(result["sql"])
    except Exception as e:
        result["error"] = f"SQL error: {e}\n\nSQL:\n{result['sql']}"

    return result


# ── CLI test harness ───────────────────────────────────────────────
if __name__ == "__main__":
    TEST_QUERIES = [
        # Domain 1 — single table
        "What is the breakdown of assets by equipment class?",
        # Domain 1 — join
        "Which assets have the most critical outages?",
        # Domain 1 — join + filter
        "Show all P1 emergency cases and which assets they belong to",
        # Domain 2 — single table
        "Which campaigns have the highest budget?",
        # Domain 2 — join
        "Which engineers own the most active campaigns?",
        # Domain 2 — 3-table join
        "What product families are most involved in delayed maintenance events?",
        # Cross-domain
        "Which asset equipment classes appear most in critical severity outages and have open service cases?",
    ]

    for q in TEST_QUERIES:
        print("\n" + "═" * 65)
        print(f"Q: {q}")
        print("═" * 65)

        out = run_query(q)

        if out["error"]:
            print(f"  ✗ ERROR: {out['error']}")
            continue

        print(f"  Explanation : {out['explanation']}")
        print(f"  Chart       : {out['chart']['type']} — "
              f"x={out['chart'].get('x')}  y={out['chart'].get('y')}")
        print(f"  SQL         :\n    {out['sql']}")
        print(f"  Rows        : {len(out['data'])}")
        if not out["data"].empty:
            print(out["data"].head(5).to_string(index=False))
