"""
semantic_inferrer_v2.py
───────────────────────
Replaces generate_ontology_draft.py entirely.
Feeds actual column stats to Claude Haiku — no hardcoded edges.

Per Hrishikesh's requirement:
  LLM infers semantics from:
    - column names
    - data types
    - sample values
    - value distributions (top values + frequency %)
    - null density
    - distinct ratio

Outputs:
  ontology.json        — nodes + edges + slot_context (LLM-generated)
  semantic_graph.png   — visualization
"""

import os
import json
import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ── Build column stat summary for LLM prompt ──────────────────────
def build_table_summary(table_key: str, profile: dict) -> str:
    lines = [f"TABLE: {table_key}  ({profile['total_rows']:,} rows)"]
    lines.append("COLUMNS:")
    for col, info in profile["columns"].items():
        line = (
            f"  {col}"
            f"  [type={info['semantic_type']}"
            f"  distinct_ratio={info['distinct_ratio']}"
            f"  null_density={info['null_density']}]"
        )
        if info["top_values"]:
            top = ", ".join(
                f"{v['value']}({v['pct']}%)" for v in info["top_values"][:3]
            )
            line += f"  top_values=[{top}]"
        if info["sample_values"]:
            line += f"  samples={info['sample_values'][:3]}"
        lines.append(line)
    return "\n".join(lines)


# ── LLM semantic inference per table ─────────────────────────────
SYSTEM_PROMPT = """You are a semantic data modeling expert specializing in industrial energy systems.

Given a table's column statistics (column names, types, distinct ratios, null densities, value distributions, sample values), you infer:
1. The semantic entity name this table represents (e.g. "PowerAsset", "GridOutage", "ServiceTicket")
2. A one-sentence business description
3. For each column: its semantic meaning in the energy/industrial domain

Return ONLY valid JSON, no markdown:
{
  "entity_label": "PascalCase entity name",
  "description": "one sentence business description",
  "columns": {
    "COLUMN_NAME": {
      "semantic_name": "human readable slot name",
      "business_meaning": "what this column represents in the energy domain"
    }
  }
}"""

def infer_table_semantics(table_key: str, profile: dict) -> dict:
    summary = build_table_summary(table_key, profile)

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": summary}],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"LLM returned non-JSON for {table_key}:\n{raw}")


# ── LLM relationship inference ────────────────────────────────────
REL_SYSTEM_PROMPT = """You are a semantic data modeling expert for industrial energy systems.

Given two tables' column statistics and a known structural join key between them, infer the most semantically accurate relationship verb in the energy domain.

Return ONLY valid JSON:
{
  "relationship": "VERB_IN_CAPS",
  "description": "one sentence explaining this relationship in the energy domain",
  "cardinality": "many_to_one | one_to_many | many_to_many | one_to_one"
}"""

def infer_relationship_semantics(
    from_table: str, from_col: str,
    to_table: str, to_col: str,
    from_profile: dict, to_profile: dict
) -> dict:
    from_summary = build_table_summary(from_table, from_profile)
    to_summary   = build_table_summary(to_table, to_profile)

    prompt = f"""FROM TABLE (references the other):
{from_summary}

TO TABLE (referenced):
{to_summary}

JOIN: {from_table}.{from_col} = {to_table}.{to_col}

What is the semantic relationship from {from_table} TO {to_table}?"""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        system=REL_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"relationship": "RELATES_TO", "description": "structural join", "cardinality": "many_to_one"}


# ── Build slot_context for query agent ───────────────────────────
def build_slot_context(all_profiles: dict) -> dict:
    slot_context = {}
    for table_key, profile in all_profiles.items():
        slot_context[table_key] = {}
        for col, info in profile["columns"].items():
            if info["semantic_type"] == "enum":
                slot_context[table_key][col] = {
                    "type":   "enum",
                    "values": [v["value"] for v in info["top_values"]],
                }
            else:
                slot_context[table_key][col] = {
                    "type": info["semantic_type"],
                }
    return slot_context


# ── Build ontology.json ───────────────────────────────────────────
def build_ontology(all_profiles: dict, relationships: list) -> dict:
    print("\nInferring table semantics via Haiku...")
    nodes = []
    table_semantics = {}

    for table_key, profile in all_profiles.items():
        print(f"  → {table_key}")
        sem = infer_table_semantics(table_key, profile)
        table_semantics[table_key] = sem
        nodes.append({
            "id":          table_key,
            "label":       sem["entity_label"],
            "table":       profile["table"],
            "schema":      profile["schema"],
            "description": sem["description"],
            "column_semantics": sem["columns"],
        })

    print("\nInferring relationship semantics via Haiku...")
    edges = []
    for r in relationships:
        if r.get("confidence") == "conceptual":
            edges.append({
                "from_table":   r["from_table"],
                "to_table":     r["to_table"],
                "from_col":     r["from_col"],
                "to_col":       r["to_col"],
                "from_label":   table_semantics.get(r["from_table"], {}).get("entity_label", r["from_table"]),
                "to_label":     table_semantics.get(r["to_table"], {}).get("entity_label", r["to_table"]),
                "relationship": "CONTEXTUALLY_LINKED",
                "description":  r.get("note", "Conceptual cross-domain link"),
                "cardinality":  "many_to_many",
                "cross_db":     True,
                "confidence":   "conceptual",
            })
            continue

        print(f"  → {r['from_table']}.{r['from_col']} → {r['to_table']}.{r['to_col']}")
        rel_sem = infer_relationship_semantics(
            r["from_table"], r["from_col"],
            r["to_table"],   r["to_col"],
            all_profiles.get(r["from_table"], {}),
            all_profiles.get(r["to_table"], {}),
        )
        edges.append({
            "from_table":   r["from_table"],
            "to_table":     r["to_table"],
            "from_col":     r["from_col"],
            "to_col":       r["to_col"],
            "from_label":   table_semantics.get(r["from_table"], {}).get("entity_label", r["from_table"]),
            "to_label":     table_semantics.get(r["to_table"], {}).get("entity_label", r["to_table"]),
            "relationship": rel_sem["relationship"],
            "description":  rel_sem["description"],
            "cardinality":  rel_sem["cardinality"],
            "cross_db":     r["cross_db"],
            "confidence":   r["confidence"],
        })
        print(f"     → [{rel_sem['relationship']}]  ({rel_sem['cardinality']})")

    slot_context = build_slot_context(all_profiles)

    return {
        "metadata": {
            "version":    "2.0",
            "method":     "LLM-inferred from column stats (null_density, distinct_ratio, value_distributions, sample_values)",
            "model":      "claude-haiku-4-5",
            "domains":    ["domain1 - Operational & Power Grid", "domain2 - Energy Service Campaigns"],
        },
        "nodes":        nodes,
        "edges":        edges,
        "slot_context": slot_context,
    }


# ── Visualization ─────────────────────────────────────────────────
def visualize(ontology: dict, output_path: str = "semantic_graph_v2.png"):
    G = nx.DiGraph()

    SCHEMA_COLORS  = {"domain1": "#D4E6F1", "domain2": "#D5F5E3"}
    BORDER_COLORS  = {"domain1": "#1A5276", "domain2": "#0E6655"}

    for node in ontology["nodes"]:
        G.add_node(node["id"], **node)

    seen = set()
    for edge in ontology["edges"]:
        key = (edge["from_table"], edge["to_table"])
        if key in seen:
            continue
        seen.add(key)
        G.add_edge(edge["from_table"], edge["to_table"], **edge)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    pos = {
    # Domain 1 — left side, top to bottom by centrality
    "domain1.ASSET_VW":                  (0.35, 0.85),
    "domain1.OUTAGE__C_VW":              (0.10, 0.50),
    "domain1.CASE_VW":                   (0.35, 0.50),
    "domain1.SERVICE_WINDOW_C_VW_TEST":  (0.60, 0.50),
    # Domain 2 — right side
    "domain2.USER_VW":                   (0.90, 0.85),
    "domain2.CAMPAIGN_VW":               (0.90, 0.50),
    "domain2.PRODUCT2_VW":               (0.90, 0.15),
    "domain2.EVENT_VW":                  (0.65, 0.15),}

    node_colors  = [SCHEMA_COLORS.get(G.nodes[n].get("schema",""), "#F0F0F0") for n in G.nodes]
    border_colors= [BORDER_COLORS.get(G.nodes[n].get("schema",""), "#555") for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=4000, edgecolors=border_colors, linewidths=1.8)

    labels = {n: f"{G.nodes[n].get('label', n)}\n({G.nodes[n].get('table', '')})" for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7.5, font_weight="bold")

    cross_edges = [(u,v) for u,v,d in G.edges(data=True) if d.get("cross_db")]
    same_edges  = [(u,v) for u,v,d in G.edges(data=True) if not d.get("cross_db")]

    nx.draw_networkx_edges(G, pos, edgelist=same_edges, ax=ax,
                           edge_color="#2E86C1", width=2.0, arrows=True, arrowsize=20,
                           connectionstyle="arc3,rad=0.1",
                           min_source_margin=45, min_target_margin=45)
    nx.draw_networkx_edges(G, pos, edgelist=cross_edges, ax=ax,
                           edge_color="#E74C3C", width=2.0, style="dashed", arrows=True, arrowsize=20,
                           connectionstyle="arc3,rad=0.2",
                           min_source_margin=45, min_target_margin=45)

    edge_labels = {(u,v): d.get("relationship","") for u,v,d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_size=7, font_color="#1C2833",
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.9))

    handles = [
        mpatches.Patch(color="#D4E6F1", label="domain1 — Power Grid",    edgecolor="#1A5276", linewidth=1),
        mpatches.Patch(color="#D5F5E3", label="domain2 — Campaigns",     edgecolor="#0E6655", linewidth=1),
        mpatches.Patch(color="#2E86C1", label="same-DB relationship"),
        mpatches.Patch(color="#E74C3C", label="cross-DB relationship"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_title(
        "Siemens Energy — Semantic Ontology Graph\n"
        "(LLM-inferred from column stats: null density · distinct ratio · value distributions · sample values)",
        fontsize=12, pad=18
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ {output_path} saved")


# ── get_semantic_context for query agent ──────────────────────────
def get_semantic_context(ontology: dict) -> str:
    lines = ["SEMANTIC ONTOLOGY CONTEXT", "=" * 60]
    lines.append("\nENTITIES:")
    for node in ontology["nodes"]:
        lines.append(f"  {node['label']} (table: {node['id']})")
        lines.append(f"    {node['description']}")

    lines.append("\nRELATIONSHIPS:")
    for edge in ontology["edges"]:
        tag = "[cross-DB]" if edge["cross_db"] else "[same-DB] "
        lines.append(
            f"  {tag} {edge['from_label']} --[{edge['relationship']}]--> {edge['to_label']}"
            f"  (JOIN ON {edge['from_col']} = {edge['to_col']})"
        )

    lines.append("\nSLOT CONTEXT (enums — use exact values in SQL WHERE clauses):")
    for table_key, slots in ontology.get("slot_context", {}).items():
        enum_slots = {k: v for k, v in slots.items() if v.get("type") == "enum"}
        if enum_slots:
            lines.append(f"  {table_key}:")
            for slot, info in enum_slots.items():
                lines.append(f"    {slot}: {info['values']}")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("Loading column_profiles.json...")
    with open("column_profiles.json") as f:
        all_profiles = json.load(f)

    print("Loading relationships.json...")
    with open("relationships.json") as f:
        relationships = json.load(f)

    ontology = build_ontology(all_profiles, relationships)

    with open("ontology.json", "w") as f:
        json.dump(ontology, f, indent=2)
    print("✅ ontology.json written")

    print("\nBuilding semantic graph...")
    visualize(ontology)

    print("\n── Semantic summary ──────────────────────────────────────")
    for node in ontology["nodes"]:
        print(f"  {node['label']:<25} ← {node['id']}")
    print()
    for edge in ontology["edges"]:
        tag = "🔀" if edge["cross_db"] else "  "
        print(f"  {tag} {edge['from_label']} --[{edge['relationship']}]--> {edge['to_label']}")


if __name__ == "__main__":
    main()
