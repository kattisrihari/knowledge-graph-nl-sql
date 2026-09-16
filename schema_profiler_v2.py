"""
schema_profiler_v2.py
─────────────────────
Profiles domain1.db and domain2.db using SQL-pushed stats:

  VARCHAR/TEXT columns:
    - row_count, null_count, null_percent, distinct_count, distinct_ratio
    - top 10 value distribution (value, frequency, percentage)
    - 10 sample non-null values

  NUMERIC columns (REAL / INTEGER):
    - same null + distinct stats
    - min, max, avg

All stats computed via SQL — no Python-side row fetching.

Outputs:
  column_profiles.json  — full stats per table per column
  relationships.json    — FK candidates via structural inference
"""

import json
import sqlite3
from itertools import combinations

DB_MAP = {
    "domain1": {
        "path":   "domain1.db",
        "tables": ["ASSET_VW", "OUTAGE__C_VW", "CASE_VW", "SERVICE_WINDOW_C_VW_TEST"],
    },
    "domain2": {
        "path":   "domain2.db",
        "tables": ["USER_VW", "PRODUCT2_VW", "CAMPAIGN_VW", "EVENT_VW"],
    },
}

NUMERIC_TYPES = {"REAL", "INTEGER", "FLOAT", "DOUBLE", "NUMERIC", "INT", "BIGINT"}


# ── Helpers ────────────────────────────────────────────────────────
def is_numeric(sqlite_type: str) -> bool:
    return sqlite_type.upper().split("(")[0].strip() in NUMERIC_TYPES

def infer_semantic_type(col: str, sqlite_type: str,
                        distinct_ratio: float, distinct_count: int) -> str:
    col_up = col.upper()
    if is_numeric(sqlite_type):
        if any(kw in col_up for kw in ["COST", "EUR", "USD", "MWH", "MW",
                                        "BUDGET", "CAPACITY", "RATE", "PCT",
                                        "PERCENT", "PRICE", "AMOUNT"]):
            return "metric"
        if any(kw in col_up for kw in ["DAYS", "HRS", "HOURS", "YEARS",
                                        "COUNT", "QTY", "STOCK"]):
            return "measure"
        return "numeric"
    if distinct_ratio < 0.05 and distinct_count <= 20:
        return "enum"
    if distinct_ratio > 0.90:
        return "identifier" if col_up.endswith(("_ID", "_NUMBER", "_CODE", "_REF")) else "free_text"
    if any(kw in col_up for kw in ["DATE", "TIME", "_AT", "_ON"]):
        return "datetime"
    return "text"


# ── SQL-based column profiler ──────────────────────────────────────
def profile_column_text(conn, table: str, col: str) -> dict:
    """VARCHAR/TEXT stats — all computed in SQL."""
    cur = conn.cursor()
    c = f'"{col}"'

    # Base stats
    cur.execute(f"""
        SELECT
            COUNT(*)                                                          AS row_count,
            SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)                    AS null_count,
            ROUND(
                100.0 * SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)
                / COUNT(*), 2
            )                                                                 AS null_percent,
            COUNT(DISTINCT {c})                                               AS distinct_count,
            ROUND(
                1.0 * COUNT(DISTINCT {c}) / NULLIF(COUNT(*), 0), 4
            )                                                                 AS distinct_ratio
        FROM {table}
    """)
    row = cur.fetchone()
    row_count, null_count, null_percent, distinct_count, distinct_ratio = row

    # Top 10 value distribution
    cur.execute(f"""
        SELECT
            {c}                                                     AS value,
            COUNT(*)                                                AS frequency,
            ROUND(
                100.0 * COUNT(*) /
                (SELECT COUNT(*) FROM {table}), 2
            )                                                       AS percentage
        FROM {table}
        WHERE {c} IS NOT NULL
        GROUP BY {c}
        ORDER BY frequency DESC
        LIMIT 10
    """)
    top_values = [
        {"value": str(r[0]), "frequency": r[1], "percentage": r[2]}
        for r in cur.fetchall()
    ]

    # 10 sample non-null values
    cur.execute(f"""
        SELECT {c} FROM {table}
        WHERE {c} IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 10
    """)
    sample_values = [str(r[0]) for r in cur.fetchall()]

    return {
        "row_count":      row_count,
        "null_count":     null_count,
        "null_percent":   null_percent,
        "null_density":   round((null_count or 0) / row_count, 4) if row_count else 0,
        "distinct_count": distinct_count,
        "distinct_ratio": distinct_ratio,
        "top_values":     top_values,
        "sample_values":  sample_values,
    }


def profile_column_numeric(conn, table: str, col: str) -> dict:
    """NUMERIC stats — null/distinct + min/max/avg, all in SQL."""
    cur = conn.cursor()
    c = f'"{col}"'

    cur.execute(f"""
        SELECT
            COUNT(*)                                                          AS row_count,
            SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)                    AS null_count,
            ROUND(
                100.0 * SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)
                / COUNT(*), 2
            )                                                                 AS null_percent,
            COUNT(DISTINCT {c})                                               AS distinct_count,
            ROUND(
                1.0 * COUNT(DISTINCT {c}) / NULLIF(COUNT(*), 0), 4
            )                                                                 AS distinct_ratio,
            MIN({c})                                                          AS min_val,
            MAX({c})                                                          AS max_val,
            ROUND(AVG({c}), 2)                                               AS avg_val
        FROM {table}
    """)
    row = cur.fetchone()
    row_count, null_count, null_percent, distinct_count, distinct_ratio, \
        min_val, max_val, avg_val = row

    # Top 10 distribution (useful for INTEGER enums like YEARS_EXPERIENCE)
    cur.execute(f"""
        SELECT
            {c}                                                     AS value,
            COUNT(*)                                                AS frequency,
            ROUND(
                100.0 * COUNT(*) /
                (SELECT COUNT(*) FROM {table}), 2
            )                                                       AS percentage
        FROM {table}
        WHERE {c} IS NOT NULL
        GROUP BY {c}
        ORDER BY frequency DESC
        LIMIT 10
    """)
    top_values = [
        {"value": str(r[0]), "frequency": r[1], "percentage": r[2]}
        for r in cur.fetchall()
    ]

    # 10 sample values
    cur.execute(f"""
        SELECT {c} FROM {table}
        WHERE {c} IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 10
    """)
    sample_values = [str(r[0]) for r in cur.fetchall()]

    return {
        "row_count":      row_count,
        "null_count":     null_count,
        "null_percent":   null_percent,
        "null_density":   round((null_count or 0) / row_count, 4) if row_count else 0,
        "distinct_count": distinct_count,
        "distinct_ratio": distinct_ratio,
        "min":            min_val,
        "max":            max_val,
        "avg":            avg_val,
        "top_values":     top_values,
        "sample_values":  sample_values,
    }


# ── Table profiler ─────────────────────────────────────────────────
def profile_table(db_path: str, schema: str, table: str) -> dict:
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    cur.execute(f"SELECT COUNT(*) FROM {table}")
    total_rows = cur.fetchone()[0]

    cur.execute(f"PRAGMA table_info({table})")
    pragma  = cur.fetchall()
    columns = [(row[1], row[2]) for row in pragma]   # (name, type)

    col_profiles = {}
    for col, sqlite_type in columns:
        if is_numeric(sqlite_type):
            stats = profile_column_numeric(conn, table, col)
        else:
            stats = profile_column_text(conn, table, col)

        sem_type = infer_semantic_type(
            col, sqlite_type,
            stats["distinct_ratio"], stats["distinct_count"]
        )
        col_profiles[col] = {
            "sqlite_type":  sqlite_type,
            "semantic_type": sem_type,
            **stats,
        }

    conn.close()
    return {
        "table":      table,
        "schema":     schema,
        "table_key":  f"{schema}.{table}",
        "total_rows": total_rows,
        "columns":    col_profiles,
    }


# ── Relationship inference (structural) ───────────────────────────
def is_id_col(col: str) -> bool:
    return col.upper().endswith(("_ID", "_KEY", "_CODE", "_REF"))

def is_pk(profile: dict, col: str) -> bool:
    """True PK: distinct_ratio > 0.95 — nearly every row is unique."""
    return profile["columns"].get(col, {}).get("distinct_ratio", 0) > 0.95

SYNONYMS = {
    "OWNER_ID": "USER_ID",
    "USER_ID":  "OWNER_ID",
}

def infer_relationships(all_profiles: dict) -> list:
    relationships = []
    table_keys = list(all_profiles.keys())

    for t_a, t_b in combinations(table_keys, 2):
        cols_a   = all_profiles[t_a]["columns"]
        cols_b   = all_profiles[t_b]["columns"]
        schema_a = all_profiles[t_a]["schema"]
        schema_b = all_profiles[t_b]["schema"]

        for col_a in cols_a:
            for col_b in cols_b:
                col_a_up = col_a.upper()
                col_b_up = col_b.upper()

                # Rule 1 — exact name match OR known synonym
                if col_a_up != col_b_up:
                    if SYNONYMS.get(col_a_up) != col_b_up:
                        continue

                # Rule 2 — must be ID-like
                if not is_id_col(col_a):
                    continue

                # Rule 3 — exactly one side is a true PK
                a_is_pk = is_pk(all_profiles[t_a], col_a)
                b_is_pk = is_pk(all_profiles[t_b], col_b)

                if a_is_pk and not b_is_pk:
                    pk_table, pk_col = t_a, col_a
                    fk_table, fk_col = t_b, col_b
                elif b_is_pk and not a_is_pk:
                    pk_table, pk_col = t_b, col_b
                    fk_table, fk_col = t_a, col_a
                else:
                    continue

                cross_db = schema_a != schema_b
                relationships.append({
                    "from_table":    fk_table,
                    "from_col":      fk_col,
                    "to_table":      pk_table,
                    "to_col":        pk_col,
                    "overlap_ratio": round(
                        all_profiles[pk_table]["columns"][pk_col]["distinct_ratio"], 3
                    ),
                    "confidence":    "high",
                    "cross_db":      cross_db,
                })
                tag  = "🔀 cross-DB" if cross_db else "   same-DB"
                print(f"  🟢 {tag}  {fk_table}.{fk_col} → {pk_table}.{pk_col}")

    return relationships


# ── Main ──────────────────────────────────────────────────────────
def main():
    all_profiles = {}

    for schema, cfg in DB_MAP.items():
        print(f"\nProfiling {schema} ({cfg['path']}) via SQL...")
        for table in cfg["tables"]:
            key     = f"{schema}.{table}"
            profile = profile_table(cfg["path"], schema, table)
            all_profiles[key] = profile

            enums   = sum(1 for c in profile["columns"].values()
                          if c["semantic_type"] == "enum")
            nulls   = sum(1 for c in profile["columns"].values()
                          if c["null_count"] and c["null_count"] > 0)
            metrics = sum(1 for c in profile["columns"].values()
                          if c["semantic_type"] in ("metric", "measure", "numeric"))

            print(f"  ✓ {table}: {profile['total_rows']:,} rows | "
                  f"{len(profile['columns'])} cols | "
                  f"{enums} enums | {metrics} numeric | {nulls} nullable")

    print("\nInferring relationships (structural, SQL-based)...")
    relationships = infer_relationships(all_profiles)
    print(f"  ✓ {len(relationships)} relationships found")

    with open("column_profiles.json", "w") as f:
        json.dump(all_profiles, f, indent=2, default=str)
    print("\n✅ column_profiles.json written")

    with open("relationships.json", "w") as f:
        json.dump(relationships, f, indent=2)
    print("✅ relationships.json written")

    # Print sample profile for one column to verify SQL stats
    print("\n── Sample column profile (ASSET_VW.EQUIPMENT_CLASS) ─────")
    sample = all_profiles["domain1.ASSET_VW"]["columns"]["EQUIPMENT_CLASS"]
    for k, v in sample.items():
        if k != "top_values":
            print(f"  {k}: {v}")
    print("  top_values:")
    for tv in sample["top_values"][:5]:
        print(f"    {tv}")


if __name__ == "__main__":
    main()