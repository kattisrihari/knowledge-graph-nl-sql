"""
Local Schema Profiler (SQLite)
- Loads CSVs into operations.db and quality.db
- Profiles all tables: columns, types, sample values
- Infers cross-DB FK relationships via name match + value overlap
- Outputs: relationships.json, schema_context.json
"""

import os
import json
import re
import sqlite3
import pandas as pd
from itertools import combinations

DATA_DIR = "data"
SAMPLE_SIZE = 200
OVERLAP_THRESHOLD = 0.3

DB_MAP = {
    "operations": {
        "path": "operations.db",
        "tables": ["machines", "maintenance_logs", "parts_inventory"],
    },
    "quality": {
        "path": "quality.db",
        "tables": ["production_batches", "defect_reports"],
    },
}


# ── Step 1: Load CSVs → SQLite ─────────────────────────────────────────────────
def load_csvs():
    for schema, cfg in DB_MAP.items():
        conn = sqlite3.connect(cfg["path"])
        for table in cfg["tables"]:
            csv_path = os.path.join(DATA_DIR, f"{table}.csv")
            if not os.path.exists(csv_path):
                print(f"  ⚠ Missing {csv_path} — skipping")
                continue
            df = pd.read_csv(csv_path)
            df.to_sql(table, conn, if_exists="replace", index=False)
            print(f"  ✓ Loaded {schema}.{table}: {len(df):,} rows")
        conn.close()


# ── Step 2: Fetch metadata + samples ──────────────────────────────────────────
def fetch_metadata():
    """Returns { 'schema.table': { col: {type, samples} } }"""
    metadata = {}

    for schema, cfg in DB_MAP.items():
        conn = sqlite3.connect(cfg["path"])
        cur = conn.cursor()

        for table in cfg["tables"]:
            key = f"{schema}.{table}"

            # Column names + types via PRAGMA
            cur.execute(f"PRAGMA table_info({table})")
            pragma_rows = cur.fetchall()  # (cid, name, type, notnull, dflt, pk)

            cols = {}
            for row in pragma_rows:
                col_name = row[1]
                col_type = row[2] if row[2] else "TEXT"
                cols[col_name] = {"type": col_type, "samples": []}

            # Sample rows
            col_names = list(cols.keys())
            cur.execute(f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT {SAMPLE_SIZE}")
            rows = cur.fetchall()
            for row in rows:
                for i, val in enumerate(row):
                    cols[col_names[i]]["samples"].append(str(val) if val is not None else None)

            metadata[key] = cols

        conn.close()

    return metadata


# ── Step 3: Row counts ─────────────────────────────────────────────────────────
def fetch_row_counts():
    counts = {}
    for schema, cfg in DB_MAP.items():
        conn = sqlite3.connect(cfg["path"])
        cur = conn.cursor()
        for table in cfg["tables"]:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            counts[f"{schema}.{table}"] = cur.fetchone()[0]
        conn.close()
    return counts


# ── Step 4: Infer relationships ────────────────────────────────────────────────
def value_overlap_ratio(samples_a, samples_b):
    set_a = set(v for v in samples_a if v is not None)
    set_b = set(v for v in samples_b if v is not None)
    if not set_a:
        return 0.0
    return len(set_a & set_b) / len(set_a)


def is_id_column(col_name):
    return any(col_name.lower().endswith(s) for s in ["_id", "_key", "_code"])


def infer_relationships(metadata):
    relationships = []
    table_keys = list(metadata.keys())

    for t_a, t_b in combinations(table_keys, 2):
        cols_a = metadata[t_a]
        cols_b = metadata[t_b]

        for col_a, info_a in cols_a.items():
            for col_b, info_b in cols_b.items():
                if col_a.lower() != col_b.lower():
                    continue
                if not is_id_column(col_a):
                    continue

                samples_a = [s for s in info_a["samples"] if s]
                samples_b = [s for s in info_b["samples"] if s]
                if not samples_a or not samples_b:
                    continue

                overlap_ab = value_overlap_ratio(samples_a, samples_b)
                overlap_ba = value_overlap_ratio(samples_b, samples_a)
                overlap = max(overlap_ab, overlap_ba)

                if overlap < OVERLAP_THRESHOLD:
                    continue

                # PK side = fewer unique sampled values (dimension/master table)
                unique_a = len(set(samples_a))
                unique_b = len(set(samples_b))

                if unique_a <= unique_b:
                    pk_table, pk_col = t_a, col_a
                    fk_table, fk_col = t_b, col_b
                else:
                    pk_table, pk_col = t_b, col_b
                    fk_table, fk_col = t_a, col_a

                relationships.append({
                    "from_table":    fk_table,
                    "from_col":      fk_col,
                    "to_table":      pk_table,
                    "to_col":        pk_col,
                    "overlap_ratio": round(overlap, 3),
                    "confidence":    "high" if overlap > 0.6 else "medium",
                    "cross_db":      fk_table.split(".")[0] != pk_table.split(".")[0],
                })
                tag = "🔀 cross-DB" if relationships[-1]["cross_db"] else "   same-DB"
                conf = "🟢" if overlap > 0.6 else "🟡"
                print(f"  {conf} {tag}  {fk_table}.{fk_col} → {pk_table}.{pk_col}  (overlap={overlap:.2f})")

    return relationships


# ── Step 5: Build schema_context for embedding later ──────────────────────────
def build_schema_context(metadata, relationships, row_counts):
    schema_context = {}

    for table_key, cols in metadata.items():
        col_descriptions = []
        for col_name, info in cols.items():
            samples = [v for v in info["samples"][:5] if v is not None]
            col_descriptions.append({
                "column":        col_name,
                "type":          info["type"],
                "sample_values": samples,
            })

        fks         = [r for r in relationships if r["from_table"] == table_key]
        referenced  = [r for r in relationships if r["to_table"]   == table_key]

        fk_strs  = [f"{r['from_col']} → {r['to_table']}.{r['to_col']}" for r in fks]
        ref_strs = [f"{r['from_table']}.{r['from_col']}" for r in referenced]

        col_names = [c["column"] for c in col_descriptions]
        summary = (
            f"Table {table_key} has {len(col_descriptions)} columns: {', '.join(col_names)}. "
            f"Row count: {row_counts.get(table_key, 'unknown')}. "
            f"Foreign keys: {'; '.join(fk_strs) if fk_strs else 'none'}. "
            f"Referenced by: {'; '.join(ref_strs) if ref_strs else 'none'}."
        )

        schema_context[table_key] = {
            "table":          table_key,
            "row_count":      row_counts.get(table_key, 0),
            "columns":        col_descriptions,
            "foreign_keys":   fk_strs,
            "referenced_by":  ref_strs,
            "text_summary":   summary,
        }

    return schema_context


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("═" * 55)
    print("Step 1 — Loading CSVs into SQLite")
    print("═" * 55)
    load_csvs()

    print("\n" + "═" * 55)
    print("Step 2 — Fetching schema metadata + samples")
    print("═" * 55)
    metadata = fetch_metadata()
    print(f"  ✓ Profiled {len(metadata)} tables")

    print("\n" + "═" * 55)
    print("Step 3 — Row counts")
    print("═" * 55)
    row_counts = fetch_row_counts()
    for k, v in row_counts.items():
        print(f"  {k}: {v:,}")

    print("\n" + "═" * 55)
    print("Step 4 — Inferring relationships")
    print("═" * 55)
    relationships = infer_relationships(metadata)
    print(f"\n  ✓ {len(relationships)} relationships inferred")

    print("\n" + "═" * 55)
    print("Step 5 — Writing outputs")
    print("═" * 55)
    schema_context = build_schema_context(metadata, relationships, row_counts)

    with open("relationships.json", "w") as f:
        json.dump(relationships, f, indent=2)
    print("  ✓ relationships.json")

    with open("schema_context.json", "w") as f:
        json.dump(schema_context, f, indent=2, default=str)
    print("  ✓ schema_context.json")

    print("\n══ Relationship summary ══════════════════════════════")
    for r in relationships:
        tag  = "cross-DB" if r["cross_db"] else "same-DB "
        conf = "🟢" if r["confidence"] == "high" else "🟡"
        print(f"  {conf} [{tag}] {r['from_table']}.{r['from_col']}"
              f" → {r['to_table']}.{r['to_col']}")
    print()


if __name__ == "__main__":
    main()
