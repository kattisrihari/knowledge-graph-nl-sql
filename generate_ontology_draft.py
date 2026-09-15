"""
generate_ontology_draft.py
─────────────────────────
Parses siemens_schema.yaml (schema-automator output) and produces
ontology_draft.json — a human-annotatable semantic ontology draft.

Annotators: search "AUTO_PROPOSED" in the JSON, review each edge,
change annotation_status to "REVIEWED" when confirmed or corrected.

Outputs: ontology_draft.json
"""

import json
import yaml
import re
from pathlib import Path

SCHEMA_FILE = "siemens_schema.yaml"
OUTPUT_FILE = "ontology_draft.json"

# ── Schema assignment (which DB each class lives in) ──────────────────────────
# Derived from our DB design — not in the YAML itself
SCHEMA_ASSIGNMENT = {
    "machines":           "operations",
    "maintenance_logs":   "operations",
    "parts_inventory":    "operations",
    "production_batches": "quality",
    "defect_reports":     "quality",
}

# ── Semantic label mapping (table → business entity name) ─────────────────────
SEMANTIC_LABELS = {
    "machines":           "Machine",
    "maintenance_logs":   "MaintenanceEvent",
    "parts_inventory":    "Part",
    "production_batches": "ProductionBatch",
    "defect_reports":     "DefectReport",
}

# ── Node descriptions ──────────────────────────────────────────────────────────
NODE_DESCRIPTIONS = {
    "machines":           "Physical manufacturing asset deployed on the factory floor (CNC mills, robotic arms, welding units, etc.)",
    "maintenance_logs":   "Record of a maintenance intervention performed on a machine, including issue type, downtime, and repair cost",
    "parts_inventory":    "Spare part or component stocked in a warehouse and compatible with one or more machines",
    "production_batches": "A discrete production run executed by a machine, tracking units produced, defect rate, and shift",
    "defect_reports":     "Quality defect identified during or after a production batch, linked to a specific part and root cause",
}

# ── Auto-proposed edge definitions ────────────────────────────────────────────
# join_key: the shared column that signals this relationship
# relationship: auto-proposed verb — annotator should confirm
# description: plain English — annotator should enrich
# cardinality: auto-inferred from data design
# include_as_attribute: whether join_key is also kept as a slot on both nodes

EDGE_DEFINITIONS = [
    {
        "from_table":           "operations.maintenance_logs",
        "to_table":             "operations.machines",
        "join_key":             "machine_id",
        "relationship":         "LOGGED_FOR",
        "description":          "A maintenance event is recorded against the machine it was performed on",
        "cardinality":          "many_to_one",
        "cross_db":             False,
        "include_as_attribute": True,
        "annotation_status":    "AUTO_PROPOSED",
        "annotation_note":      "Annotator: confirm verb. Alternatives: PERFORMED_ON, TRIGGERED_BY, RECORDED_FOR",
    },
    {
        "from_table":           "operations.parts_inventory",
        "to_table":             "operations.machines",
        "join_key":             "machine_id",
        "relationship":         "COMPATIBLE_WITH",
        "description":          "A part is stocked as compatible with or designated for a specific machine",
        "cardinality":          "many_to_one",
        "cross_db":             False,
        "include_as_attribute": True,
        "annotation_status":    "AUTO_PROPOSED",
        "annotation_note":      "Annotator: confirm verb. Alternatives: USED_IN, DESIGNATED_FOR, SERVICED_BY",
    },
    {
        "from_table":           "quality.production_batches",
        "to_table":             "operations.machines",
        "join_key":             "machine_id",
        "relationship":         "PRODUCED_BY",
        "description":          "A production batch is executed by a specific machine on the factory floor",
        "cardinality":          "many_to_one",
        "cross_db":             True,
        "include_as_attribute": True,
        "annotation_status":    "AUTO_PROPOSED",
        "annotation_note":      "Annotator: confirm verb. Alternatives: EXECUTED_BY, RUN_ON, ASSIGNED_TO",
    },
    {
        "from_table":           "quality.production_batches",
        "to_table":             "operations.machines",
        "join_key":             "plant_id",
        "relationship":         "OCCURS_AT",
        "description":          "A production batch occurs at the same plant as the machine — plant_id is also retained as a shared attribute on both nodes",
        "cardinality":          "many_to_one",
        "cross_db":             True,
        "include_as_attribute": True,
        "annotation_status":    "AUTO_PROPOSED",
        "annotation_note":      "Annotator: plant_id is a shared attribute on both Machine and ProductionBatch. This edge captures the co-location relationship. Alternatives: LOCATED_AT, BELONGS_TO_PLANT. Can be removed if deemed redundant.",
    },
    {
        "from_table":           "quality.defect_reports",
        "to_table":             "quality.production_batches",
        "join_key":             "batch_id",
        "relationship":         "BELONGS_TO",
        "description":          "A defect report is raised against the production batch in which the defect was detected",
        "cardinality":          "many_to_one",
        "cross_db":             False,
        "include_as_attribute": True,
        "annotation_status":    "AUTO_PROPOSED",
        "annotation_note":      "Annotator: confirm verb. Alternatives: DETECTED_IN, RAISED_FOR, PART_OF",
    },
    {
        "from_table":           "quality.defect_reports",
        "to_table":             "operations.parts_inventory",
        "join_key":             "part_id",
        "relationship":         "CAUSED_BY",
        "description":          "A defect is attributed to a specific part that failed or was out of specification",
        "cardinality":          "many_to_one",
        "cross_db":             True,
        "include_as_attribute": True,
        "annotation_status":    "AUTO_PROPOSED",
        "annotation_note":      "Annotator: confirm verb. Note directionality — defect report points TO the part. Alternatives: LINKED_TO_PART, INVOLVES, ORIGINATED_FROM",
    },
]


# ── Parse YAML ────────────────────────────────────────────────────────────────
def parse_schema(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Build slot context from schema ────────────────────────────────────────────
def build_slot_context(schema: dict) -> dict:
    """
    For each class, maps every slot to its type and enum values (if applicable).
    Used by the NL-to-SQL agent for exact value matching.
    """
    enums = schema.get("enums", {})
    slots = schema.get("slots", {})
    classes = schema.get("classes", {})

    # Build enum value lookup
    enum_values = {}
    for enum_name, enum_def in enums.items():
        pv = enum_def.get("permissible_values", {})
        enum_values[enum_name] = list(pv.keys())

    slot_context = {}
    for class_name, class_def in classes.items():
        schema_name = SCHEMA_ASSIGNMENT.get(class_name, "unknown")
        table_key = f"{schema_name}.{class_name}"
        slot_context[table_key] = {}

        for slot_name in class_def.get("slots", []):
            slot_def = slots.get(slot_name, {})
            slot_range = slot_def.get("range", "string")
            is_identifier = slot_def.get("identifier", False)
            example = slot_def.get("examples", [{}])
            example_val = example[0].get("value", "") if example else ""

            if slot_range in enum_values:
                slot_context[table_key][slot_name] = {
                    "type":       "enum",
                    "values":     enum_values[slot_range],
                    "example":    example_val,
                    "identifier": is_identifier,
                }
            else:
                slot_context[table_key][slot_name] = {
                    "type":       slot_range,
                    "example":    example_val,
                    "identifier": is_identifier,
                }

    return slot_context


# ── Build nodes ───────────────────────────────────────────────────────────────
def build_nodes(schema: dict, slot_context: dict) -> list:
    classes = schema.get("classes", {})
    nodes = []

    for class_name, class_def in classes.items():
        schema_name = SCHEMA_ASSIGNMENT.get(class_name, "unknown")
        table_key = f"{schema_name}.{class_name}"
        label = SEMANTIC_LABELS.get(class_name, class_name.title())

        # Key attributes = identifiers + enum slots
        ctx = slot_context.get(table_key, {})
        key_attrs = [
            col for col, info in ctx.items()
            if info.get("identifier") or info.get("type") == "enum"
        ]

        nodes.append({
            "id":             table_key,
            "label":          label,
            "table":          class_name,
            "schema":         schema_name,
            "description":    NODE_DESCRIPTIONS.get(class_name, ""),
            "key_attributes": key_attrs,
        })

    return nodes


# ── Build edges ───────────────────────────────────────────────────────────────
def build_edges(nodes: list) -> list:
    label_map = {n["id"]: n["label"] for n in nodes}
    edges = []

    for e in EDGE_DEFINITIONS:
        edge = {
            "from_table":           e["from_table"],
            "to_table":             e["to_table"],
            "from_label":           label_map.get(e["from_table"], e["from_table"]),
            "to_label":             label_map.get(e["to_table"], e["to_table"]),
            "join_key":             e["join_key"],
            "relationship":         e["relationship"],
            "description":          e["description"],
            "cardinality":          e["cardinality"],
            "cross_db":             e["cross_db"],
            "include_as_attribute": e["include_as_attribute"],
            "annotation_status":    e["annotation_status"],
            "annotation_note":      e["annotation_note"],
        }
        edges.append(edge)

    return edges


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Reading {SCHEMA_FILE}...")
    schema = parse_schema(SCHEMA_FILE)

    print("Building slot context...")
    slot_context = build_slot_context(schema)

    print("Building nodes...")
    nodes = build_nodes(schema, slot_context)

    print("Building edges...")
    edges = build_edges(nodes)

    ontology = {
        "metadata": {
            "version":        "1.0-draft",
            "source_schema":  SCHEMA_FILE,
            "domain":         "Industrial Manufacturing",
            "client":         "Siemens",
            "annotation_instructions": (
                "Search 'AUTO_PROPOSED' to find all edges needing review. "
                "Edit 'relationship' verb and 'description' as needed. "
                "Set 'annotation_status' to 'REVIEWED' when confirmed. "
                "Do not remove 'annotation_note' — keep for audit trail."
            ),
        },
        "nodes":        nodes,
        "edges":        edges,
        "slot_context": slot_context,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(ontology, f, indent=2)

    print(f"\n✅ {OUTPUT_FILE} written")
    print(f"   {len(nodes)} nodes")
    print(f"   {len(edges)} edges ({sum(1 for e in edges if e['annotation_status'] == 'AUTO_PROPOSED')} need annotation)")
    print(f"   {sum(len(v) for v in slot_context.values())} total slots with type context")
    print(f"\n── Proposed edges ────────────────────────────────────────")
    for e in edges:
        tag = "🔀" if e["cross_db"] else "  "
        print(f"  {tag} {e['from_label']} --[{e['relationship']}]--> {e['to_label']}  (via {e['join_key']})")
    print(f"\nOpen {OUTPUT_FILE} in VS Code, search AUTO_PROPOSED, review and set to REVIEWED.")


if __name__ == "__main__":
    main()
