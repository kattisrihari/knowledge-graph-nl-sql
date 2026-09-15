"""
semantic_inferrer.py
────────────────────
Reads ontology_draft.json (or ontology.json after annotation) and builds:
  - A semantic NetworkX DiGraph (entities + named relationships)
  - semantic_graph.png — visualization for demo
  - Exposes get_semantic_context() for the query agent
"""

import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ONTOLOGY_FILE = "ontology_draft.json"   # switch to ontology.json post-annotation


# ── Load ontology ─────────────────────────────────────────────────────────────
def load_ontology(path: str = ONTOLOGY_FILE) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Build semantic graph ───────────────────────────────────────────────────────
def build_semantic_graph(ontology: dict) -> nx.DiGraph:
    G = nx.DiGraph()

    # Add nodes
    for node in ontology["nodes"]:
        G.add_node(
            node["id"],
            label=node["label"],
            schema=node["schema"],
            table=node["table"],
            description=node["description"],
            key_attributes=node["key_attributes"],
        )

    # Add edges — deduplicate same from/to/join_key combos
    seen = set()
    for edge in ontology["edges"]:
        key = (edge["from_table"], edge["to_table"], edge["join_key"])
        if key in seen:
            continue
        seen.add(key)

        G.add_edge(
            edge["from_table"],
            edge["to_table"],
            relationship=edge["relationship"],
            join_key=edge["join_key"],
            cardinality=edge["cardinality"],
            cross_db=edge["cross_db"],
            description=edge["description"],
            annotation_status=edge["annotation_status"],
            from_label=edge["from_label"],
            to_label=edge["to_label"],
        )

    return G


# ── Visualization ─────────────────────────────────────────────────────────────
def visualize(G: nx.DiGraph, ontology: dict, output_path: str = "semantic_graph.png"):
    SCHEMA_COLORS = {
        "operations": "#D4E6F1",
        "quality":    "#D5F5E3",
    }
    BORDER_COLORS = {
        "operations": "#1A5276",
        "quality":    "#0E6655",
    }

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    # Fixed layout for clarity
    pos = {
        "operations.machines":           (0.5,  0.75),
        "operations.maintenance_logs":   (0.05, 0.35),
        "operations.parts_inventory":    (0.95, 0.35),
        "quality.production_batches":    (0.28, 0.05),
        "quality.defect_reports":        (0.72, 0.05),
    }

    node_colors  = [SCHEMA_COLORS.get(G.nodes[n]["schema"], "#F0F0F0") for n in G.nodes]
    border_colors = [BORDER_COLORS.get(G.nodes[n]["schema"], "#555") for n in G.nodes]

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=4500,
        edgecolors=border_colors,
        linewidths=1.8,
    )

    # Node labels — semantic label + table name
    node_labels = {
        n: f"{G.nodes[n]['label']}\n({G.nodes[n]['table']})"
        for n in G.nodes
    }
    nx.draw_networkx_labels(
        G, pos, labels=node_labels, ax=ax,
        font_size=8, font_weight="bold", font_color="#1C2833"
    )

    # Separate edges by type for styling
    cross_edges = [(u, v) for u, v, d in G.edges(data=True) if d["cross_db"]]
    same_edges  = [(u, v) for u, v, d in G.edges(data=True) if not d["cross_db"]]

    nx.draw_networkx_edges(
        G, pos, edgelist=same_edges, ax=ax,
        edge_color="#2E86C1", width=2.0,
        arrows=True, arrowsize=22,
        connectionstyle="arc3,rad=0.12",
        min_source_margin=45, min_target_margin=45,
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=cross_edges, ax=ax,
        edge_color="#E74C3C", width=2.2, style="dashed",
        arrows=True, arrowsize=22,
        connectionstyle="arc3,rad=0.18",
        min_source_margin=45, min_target_margin=45,
    )

    # Edge labels — relationship verb + join key
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        status_marker = "" if d["annotation_status"] == "REVIEWED" else " *"
        edge_labels[(u, v)] = f"{d['relationship']}{status_marker}\n({d['join_key']})"

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_size=7, font_color="#1C2833",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.9),
    )

    # Legend
    handles = [
        mpatches.Patch(color="#D4E6F1", label="operations schema", edgecolor="#1A5276", linewidth=1),
        mpatches.Patch(color="#D5F5E3", label="quality schema",    edgecolor="#0E6655", linewidth=1),
        mpatches.Patch(color="#2E86C1", label="same-DB relationship"),
        mpatches.Patch(color="#E74C3C", label="cross-DB relationship"),
        mpatches.Patch(color="white",   label="* = pending annotation", edgecolor="#BDC3C7", linewidth=1),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, framealpha=0.95)

    ax.set_title(
        "Siemens — Semantic Ontology Graph\n"
        "(auto-inferred from instance data via schema-automator + human annotation)",
        fontsize=13, pad=18
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {output_path} saved")


# ── Query agent interface ─────────────────────────────────────────────────────
def get_semantic_context(ontology: dict) -> str:
    """
    Returns a plain-text block injected into the LLM prompt.
    Covers: entities, relationships, join keys, enum values.
    """
    lines = ["SEMANTIC ONTOLOGY CONTEXT", "=" * 50]

    lines.append("\nENTITIES:")
    for node in ontology["nodes"]:
        lines.append(f"  {node['label']} (table: {node['id']})")
        lines.append(f"    {node['description']}")

    lines.append("\nRELATIONSHIPS:")
    for edge in ontology["edges"]:
        tag = "[cross-DB]" if edge["cross_db"] else "[same-DB] "
        lines.append(
            f"  {tag} {edge['from_label']} --[{edge['relationship']}]--> {edge['to_label']}"
            f"  (JOIN ON {edge['join_key']})"
        )

    lines.append("\nSLOT CONTEXT (enums only — use exact values in SQL WHERE clauses):")
    for table_key, slots in ontology["slot_context"].items():
        enum_slots = {k: v for k, v in slots.items() if v["type"] == "enum"}
        if enum_slots:
            lines.append(f"  {table_key}:")
            for slot, info in enum_slots.items():
                lines.append(f"    {slot}: {info['values']}")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading {ONTOLOGY_FILE}...")
    ontology = load_ontology()

    print("Building semantic graph...")
    G = build_semantic_graph(ontology)
    print(f"  ✓ {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Visualizing...")
    visualize(G, ontology)

    print("\nSemantic context for query agent:")
    print(get_semantic_context(ontology))

    # Annotation status summary
    edges = ontology["edges"]
    reviewed   = sum(1 for e in edges if e["annotation_status"] == "REVIEWED")
    auto       = sum(1 for e in edges if e["annotation_status"] == "AUTO_PROPOSED")
    print(f"\n── Annotation status ─────────────────────────────────")
    print(f"  REVIEWED:      {reviewed}/{len(edges)}")
    print(f"  AUTO_PROPOSED: {auto}/{len(edges)}  ← needs human review")
    if auto:
        print("  Tip: open ontology_draft.json, search AUTO_PROPOSED, confirm verbs, set to REVIEWED")


if __name__ == "__main__":
    main()
