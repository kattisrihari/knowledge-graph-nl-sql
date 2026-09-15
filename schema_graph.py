"""
Schema Graph (NetworkX)
- Builds a directed graph from relationships.json + schema_context.json
- Nodes = tables, Edges = inferred FK relationships
- find_join_path() → used by the query agent to resolve multi-hop joins
- Saves schema_graph.png for demo visualization
"""

import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Load artifacts ─────────────────────────────────────────────────────────────
def load_artifacts(
    relationships_path="relationships.json",
    schema_context_path="schema_context.json",
):
    with open(relationships_path) as f:
        relationships = json.load(f)
    with open(schema_context_path) as f:
        schema_context = json.load(f)
    return relationships, schema_context


# ── Build graph ────────────────────────────────────────────────────────────────
def build_graph(relationships, schema_context):
    G = nx.DiGraph()

    # Add nodes with metadata
    for table_key, ctx in schema_context.items():
        schema_name = table_key.split(".")[0]
        G.add_node(
            table_key,
            schema=schema_name,
            columns=[c["column"] for c in ctx["columns"]],
            row_count=ctx["row_count"],
            text_summary=ctx["text_summary"],
        )

    # Add edges with metadata
    for r in relationships:
        G.add_edge(
            r["from_table"],
            r["to_table"],
            from_col=r["from_col"],
            to_col=r["to_col"],
            overlap=r["overlap_ratio"],
            confidence=r["confidence"],
            cross_db=r.get("cross_db", False),
        )
        # Add reverse edge too — joins work both ways
        G.add_edge(
            r["to_table"],
            r["from_table"],
            from_col=r["to_col"],
            to_col=r["from_col"],
            overlap=r["overlap_ratio"],
            confidence=r["confidence"],
            cross_db=r.get("cross_db", False),
        )

    return G


# ── Core utility: find join path ───────────────────────────────────────────────
def find_join_path(G, source, target):
    """
    Returns the shortest join path between two tables as a list of
    JOIN clauses the SQL agent can use directly.

    Example:
      find_join_path(G, 'quality.defect_reports', 'operations.machines')
      → [
          "JOIN operations.parts_inventory ON quality.defect_reports.part_id = operations.parts_inventory.part_id",
          "JOIN operations.machines ON operations.parts_inventory.machine_id = operations.machines.machine_id"
        ]
    """
    try:
        path = nx.shortest_path(G, source=source, target=target)
    except nx.NetworkXNoPath:
        return None
    except nx.NodeNotFound as e:
        print(f"  ⚠ Node not found: {e}")
        return None

    join_clauses = []
    for i in range(len(path) - 1):
        t_from = path[i]
        t_to   = path[i + 1]
        edge   = G[t_from][t_to]
        join_clauses.append(
            f"JOIN {t_to} ON {t_from}.{edge['from_col']} = {t_to}.{edge['to_col']}"
        )

    return {
        "path":         path,
        "join_clauses": join_clauses,
        "hops":         len(path) - 1,
    }


def get_all_join_paths(G):
    """Returns join paths for all table pairs — fed into LLM prompt as context."""
    tables = list(G.nodes)
    all_paths = {}
    for i, src in enumerate(tables):
        for tgt in tables[i + 1:]:
            result = find_join_path(G, src, tgt)
            if result:
                all_paths[f"{src} → {tgt}"] = result
    return all_paths


def describe_graph(G):
    """Returns a plain-text summary of the graph for LLM prompt injection."""
    lines = ["SCHEMA GRAPH SUMMARY", "=" * 40]
    lines.append(f"Tables ({G.number_of_nodes()}):")
    for node, data in G.nodes(data=True):
        lines.append(f"  {node}  [{data['row_count']:,} rows]")
        lines.append(f"    columns: {', '.join(data['columns'])}")

    lines.append(f"\nRelationships ({G.number_of_edges() // 2} unique FKs):")
    seen = set()
    for u, v, data in G.edges(data=True):
        key = tuple(sorted([u, v]))
        if key in seen:
            continue
        seen.add(key)
        tag = "cross-DB" if data["cross_db"] else "same-DB"
        lines.append(f"  {u}.{data['from_col']} ↔ {v}.{data['to_col']}  [{tag}, overlap={data['overlap']}]")

    return "\n".join(lines)


# ── Visualization ──────────────────────────────────────────────────────────────
def visualize(G, output_path="schema_graph.png"):
    SCHEMA_COLORS = {
        "operations": "#D6EAF8",   # light blue
        "quality":    "#D5F5E3",   # light green
    }
    EDGE_COLORS = {
        True:  "#E74C3C",   # cross-DB → red
        False: "#95A5A6",   # same-DB  → grey
    }

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")

    # Layout — spring with fixed seed for reproducibility
    pos = nx.spring_layout(G, seed=7, k=2.5)

    # Node colors by schema
    node_colors = [
        SCHEMA_COLORS.get(G.nodes[n]["schema"], "#F0F0F0")
        for n in G.nodes
    ]

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=3200,
        edgecolors="#2C3E50",
        linewidths=1.2,
    )

    # Draw labels (short table name + row count)
    labels = {
        n: f"{n.split('.')[1]}\n({G.nodes[n]['row_count']:,})"
        for n in G.nodes
    }
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8, font_weight="bold")

    # Draw edges — deduplicate bidirectional
    seen_edges = set()
    cross_edges, same_edges = [], []
    for u, v, data in G.edges(data=True):
        key = tuple(sorted([u, v]))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        if data["cross_db"]:
            cross_edges.append((u, v))
        else:
            same_edges.append((u, v))

    nx.draw_networkx_edges(
        G, pos, edgelist=same_edges, ax=ax,
        edge_color="#7F8C8D", width=1.5,
        arrows=True, arrowsize=18,
        connectionstyle="arc3,rad=0.08",
        min_source_margin=35, min_target_margin=35,
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=cross_edges, ax=ax,
        edge_color="#E74C3C", width=2.2,
        arrows=True, arrowsize=20, style="dashed",
        connectionstyle="arc3,rad=0.15",
        min_source_margin=35, min_target_margin=35,
    )

    # Edge labels (col → col)
    edge_label_map = {}
    seen_labels = set()
    for u, v, data in G.edges(data=True):
        key = tuple(sorted([u, v]))
        if key in seen_labels:
            continue
        seen_labels.add(key)
        edge_label_map[(u, v)] = f"{data['from_col']}"

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_label_map, ax=ax,
        font_size=7, font_color="#2C3E50",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color="#D6EAF8", label="operations schema", linewidth=0.5, edgecolor="#2C3E50"),
        mpatches.Patch(color="#D5F5E3", label="quality schema",    linewidth=0.5, edgecolor="#2C3E50"),
        mpatches.Patch(color="#E74C3C", label="cross-DB join",     linewidth=0.5),
        mpatches.Patch(color="#7F8C8D", label="same-DB join",      linewidth=0.5),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title("Siemens — Schema Relationship Graph\n(auto-inferred, no ERD provided)", fontsize=13, pad=16)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Graph saved → {output_path}")


# ── Main (standalone test) ─────────────────────────────────────────────────────
def main():
    print("Loading artifacts...")
    relationships, schema_context = load_artifacts()

    print("Building graph...")
    G = build_graph(relationships, schema_context)
    print(f"  ✓ {G.number_of_nodes()} nodes, {G.number_of_edges() // 2} unique edges")

    print("\nVisualizing...")
    visualize(G)

    print("\nGraph description (fed to LLM):")
    print(describe_graph(G))

    print("\nJoin path tests:")
    tests = [
        ("quality.defect_reports", "operations.machines"),
        ("quality.defect_reports", "operations.parts_inventory"),
        ("quality.production_batches", "operations.maintenance_logs"),
    ]
    for src, tgt in tests:
        result = find_join_path(G, src, tgt)
        if result:
            print(f"\n  {src} → {tgt}  ({result['hops']} hop(s))")
            for clause in result["join_clauses"]:
                print(f"    {clause}")
        else:
            print(f"\n  ✗ No path: {src} → {tgt}")


if __name__ == "__main__":
    main()
