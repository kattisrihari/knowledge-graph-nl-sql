"""
generate_energy.py
──────────────────
Generates Siemens Energy domain data across two SQLite databases:
  domain1.db — Operational & Power Grid Service Domain
  domain2.db — Execution & Energy Service Campaigns Domain

Data quality rules applied per column:
  - Distinct ratio  : controls enum vs identifier detection
  - Null density    : realistic nulls, not zero
  - Value distribution : skewed, not uniform (weighted choices)
  - Sample values   : real Siemens domain vocabulary
"""

import os
import sqlite3
import random
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

OUT_DIR = "energy_data"
os.makedirs(OUT_DIR, exist_ok=True)

START_DATE = datetime(2021, 1, 1)
END_DATE   = datetime(2024, 12, 31)

def rand_dates(n, start=START_DATE, end=END_DATE):
    span = int((end - start).total_seconds())
    offsets = np.random.randint(0, span, size=n)
    return [start + timedelta(seconds=int(o)) for o in offsets]

def nullable(values, null_frac):
    """Randomly set null_frac fraction of values to None."""
    mask = np.random.random(len(values)) < null_frac
    return [None if mask[i] else values[i] for i in range(len(values))]


# ═══════════════════════════════════════════════════════════════════
# DOMAIN 1 — Operational & Power Grid Service Domain
# ═══════════════════════════════════════════════════════════════════

# ── Shared ID pools ────────────────────────────────────────────────
N_ASSETS   = 500
ASSET_IDS  = [f"AST-{str(i).zfill(5)}" for i in range(1, N_ASSETS + 1)]
PLANT_IDS  = [f"PLT-{str(i).zfill(3)}" for i in range(1, 41)]   # 40 plant locations globally

EQUIPMENT_CLASSES = ["Gas Turbine", "Wind Generator", "Transformer",
                     "Steam Turbine", "Solar Inverter", "Substation Switch"]
ASSET_MODELS = {
    "Gas Turbine":       ["SGT-800", "SGT-600", "SGT-A65", "SGT5-4000F"],
    "Wind Generator":    ["SWT-3.3-130", "SWT-4.0-130", "SWT-DD-130"],
    "Transformer":       ["400kV Power Transformer", "220kV Auto-Transformer", "132kV Distribution TX"],
    "Steam Turbine":     ["SST-600", "SST-800", "SST-400"],
    "Solar Inverter":    ["SINVERT 630M", "SINVERT 500S"],
    "Substation Switch": ["3AP1 FG", "3AQ1 EG", "3AV1"],
}
PLANT_REGIONS = {
    f"PLT-{str(i).zfill(3)}": random.choice(["EMEA", "APAC", "Americas"])
    for i in range(1, 41)
}

# ── 1. ASSET_VW ────────────────────────────────────────────────────
def gen_asset_vw():
    N = 500
    eq_classes = np.random.choice(
        EQUIPMENT_CLASSES, N,
        p=[0.30, 0.25, 0.20, 0.12, 0.08, 0.05]   # Gas Turbine most common
    )
    serial_nums = [f"SN-{fake.bothify('??###???##').upper()}" for _ in range(N)]
    comm_dates  = rand_dates(N, start=datetime(2010, 1, 1), end=datetime(2023, 6, 30))

    models = [random.choice(ASSET_MODELS[ec]) for ec in eq_classes]
    plant_ids = np.random.choice(PLANT_IDS, N)

    df = pd.DataFrame({
        "ASSET_ID":            ASSET_IDS,
        "PLANT_LOCATION_ID":   plant_ids,
        "SERIAL_NUMBER":       serial_nums,
        "EQUIPMENT_CLASS":     eq_classes,
        "MODEL":               models,
        "COMMISSIONING_DATE":  [d.date() for d in comm_dates],
        # dist: Running 70%, Under Maintenance 20%, Tripped 10%
        "OPERATIONAL_STATUS":  np.random.choice(
            ["Running", "Under Maintenance", "Tripped"],
            N, p=[0.70, 0.20, 0.10]
        ),
        "REGION":              [PLANT_REGIONS[p] for p in plant_ids],
        # null density ~5% — some assets not yet rated
        "RATED_CAPACITY_MW":   nullable(
            list(np.round(np.random.uniform(10, 600, N), 1)), 0.05
        ),
        "LAST_INSPECTION_DATE": nullable(
            [d.date() for d in rand_dates(N, start=datetime(2020,1,1))], 0.10
        ),
    })
    return df

# ── 2. OUTAGE__C_VW ───────────────────────────────────────────────
def gen_outage_vw():
    N = 100
    outage_ids  = [f"OUT-{str(i).zfill(6)}" for i in range(1, N+1)]
    asset_ids   = np.random.choice(ASSET_IDS, N)
    start_times = rand_dates(N)
    durations   = np.random.exponential(scale=8, size=N).clip(0.5, 72)  # hours
    end_times   = [s + timedelta(hours=float(d)) for s, d in zip(start_times, durations)]

    # ~15% ongoing — no restored time
    restored = nullable(
        [e.strftime("%Y-%m-%d %H:%M:%S") for e in end_times], 0.15
    )

    df = pd.DataFrame({
        "OUTAGE_ID":     outage_ids,
        "ASSET_ID":      asset_ids,          # FK → ASSET_VW
        "TRIP_REASON":   np.random.choice(
            ["Vibration Spike", "Thermal Overload", "Grid Frequency Drop",
             "Protection Relay Trip", "Mechanical Failure", "Electrical Fault"],
            N, p=[0.20, 0.25, 0.20, 0.15, 0.12, 0.08]
        ),
        # dist: Critical 10%, Degraded 60%, Resolved 30%
        "SEVERITY":      np.random.choice(
            ["Critical - Plant Offline", "Degraded Output", "Resolved"],
            N, p=[0.10, 0.60, 0.30]
        ),
        "START_TIME":    [s.strftime("%Y-%m-%d %H:%M:%S") for s in start_times],
        "RESTORED_TIME": restored,           # null density ~15%
        "DOWNTIME_HRS":  np.round(durations, 2),
        "LOSS_MWH":      nullable(
            list(np.round(np.random.uniform(10, 5000, N), 1)), 0.20
        ),
        "REPORTED_BY":   [f"ENG-{str(random.randint(1,50)).zfill(3)}" for _ in range(N)],
    })
    return df

# ── 3. CASE_VW ────────────────────────────────────────────────────
def gen_case_vw(outage_ids):
    N = 3000
    case_ids   = [f"CASE-{str(i).zfill(7)}" for i in range(1, N+1)]
    asset_ids  = np.random.choice(ASSET_IDS, N)
    created_at = rand_dates(N)

    # OUTAGE_ID nullable ~60% — most cases don't link to an outage
    outage_refs = nullable(
        list(np.random.choice(outage_ids, N)), 0.60
    )

    df = pd.DataFrame({
        "CASE_ID":        case_ids,
        "ASSET_ID":       asset_ids,          # FK → ASSET_VW
        "OUTAGE_ID":      outage_refs,         # FK nullable → OUTAGE__C_VW, null ~60%
        "ISSUE_CATEGORY": np.random.choice(
            ["Blade Inspection", "Control System Fault", "Oil Leakage",
             "Vibration Analysis", "Electrical Fault", "Cooling System Issue",
             "Software Bug", "Structural Crack"],
            N, p=[0.18, 0.16, 0.14, 0.13, 0.12, 0.11, 0.09, 0.07]
        ),
        # P1 5%, P2 25%, P3 55%, P4 15%
        "PRIORITY":       np.random.choice(
            ["P1 - Emergency", "P2 - Urgent", "P3 - Standard", "P4 - Low"],
            N, p=[0.05, 0.25, 0.55, 0.15]
        ),
        "STATUS":         np.random.choice(
            ["Open", "In Progress", "Resolved", "Closed", "Escalated"],
            N, p=[0.15, 0.25, 0.30, 0.25, 0.05]
        ),
        "CREATED_AT":     [d.strftime("%Y-%m-%d %H:%M:%S") for d in created_at],
        # null ~10% — not all cases assigned yet
        "ASSIGNED_TO":    nullable(
            [f"ENG-{str(random.randint(1,50)).zfill(3)}" for _ in range(N)], 0.10
        ),
        "RESOLUTION_NOTES": nullable(
            [fake.sentence(nb_words=10) for _ in range(N)], 0.35
        ),
        "ESTIMATED_COST_EUR": nullable(
            list(np.round(np.random.exponential(scale=8000, size=N).clip(200, 80000), 2)),
            0.25
        ),
    })
    return df

# ── 4. SERVICE_WINDOW_C_VW_TEST ───────────────────────────────────
def gen_service_window_vw():
    N = 200
    window_ids  = [f"WIN-{str(i).zfill(5)}" for i in range(1, N+1)]
    asset_ids   = np.random.choice(ASSET_IDS, N)
    starts      = rand_dates(N, start=datetime(2022,1,1), end=datetime(2025,6,30))
    durations_d = np.random.randint(1, 21, N)   # 1–20 days
    ends        = [s + timedelta(days=int(d)) for s, d in zip(starts, durations_d)]

    df = pd.DataFrame({
        "WINDOW_ID":       window_ids,
        "ASSET_ID":        asset_ids,           # FK → ASSET_VW
        "WINDOW_TYPE":     np.random.choice(
            ["Major Overhaul", "Borescope Inspection", "Grid Balancing",
             "Firmware Upgrade", "Predictive Maintenance"],
            N, p=[0.25, 0.30, 0.20, 0.15, 0.10]
        ),
        "APPROVED_START":  [s.date() for s in starts],
        "APPROVED_END":    [e.date() for e in ends],
        "DURATION_DAYS":   durations_d,
        # null ~20% — not yet signed off
        "SAFETY_SIGN_OFF_BY": nullable(
            [f"INSP-{str(random.randint(1,20)).zfill(3)}" for _ in range(N)], 0.20
        ),
        "APPROVAL_STATUS": np.random.choice(
            ["Approved", "Pending", "Rejected"],
            N, p=[0.70, 0.22, 0.08]
        ),
        "REGION":          np.random.choice(["EMEA", "APAC", "Americas"], N,
                                             p=[0.45, 0.30, 0.25]),
    })
    return df


# ═══════════════════════════════════════════════════════════════════
# DOMAIN 2 — Execution & Energy Service Campaigns Domain
# ═══════════════════════════════════════════════════════════════════

USER_IDS    = [f"USR-{str(i).zfill(4)}" for i in range(1, 51)]
PRODUCT_IDS = [f"PRD-{str(i).zfill(5)}" for i in range(1, 101)]
CAMPAIGN_IDS= [f"CMP-{str(i).zfill(4)}" for i in range(1, 31)]

# ── 5. USER_VW ────────────────────────────────────────────────────
def gen_user_vw():
    N = 50
    specializations = np.random.choice(
        ["Turbine Dynamics", "Grid Automation", "High Voltage Systems",
         "Blade Engineering", "Digital Controls", "Thermal Systems"],
        N, p=[0.25, 0.20, 0.20, 0.15, 0.12, 0.08]
    )
    regions = np.random.choice(["EMEA", "APAC", "Americas"], N, p=[0.45, 0.30, 0.25])
    names   = [fake.name() for _ in range(N)]

    df = pd.DataFrame({
        "USER_ID":         USER_IDS,
        "FULL_NAME":       names,
        "EMAIL":           [f"{n.lower().replace(' ','.')}{random.randint(1,99)}@siemens-energy.com"
                            for n in names],
        "SPECIALIZATION":  specializations,
        "REGION":          regions,
        "ROLE":            np.random.choice(
            ["Lead Engineer", "Project Director", "Commissioning Specialist",
             "Field Technician", "Service Manager"],
            N, p=[0.25, 0.15, 0.20, 0.25, 0.15]
        ),
        "YEARS_EXPERIENCE": np.random.randint(2, 28, N),
        # null ~8% — some contractors no phone
        "PHONE":           nullable([fake.phone_number() for _ in range(N)], 0.08),
    })
    return df

# ── 6. PRODUCT2_VW ────────────────────────────────────────────────
def gen_product2_vw():
    N = 100
    families = np.random.choice(
        ["Hardware Spare", "Software/Digital", "Service Contract"],
        N, p=[0.50, 0.30, 0.20]
    )
    product_names = {
        "Hardware Spare":    ["Compressor Blade Set v4", "Rotor Bearing Assembly",
                              "Turbine Seal Kit", "Generator Stator Winding",
                              "Control Valve Actuator", "Cooling Fan Module"],
        "Software/Digital":  ["Omnivise T3000 Control Upgrade", "SPPA-T3000 Patch v8",
                              "Predictive Analytics Module", "Digital Twin License",
                              "Cybersecurity Firmware Patch"],
        "Service Contract":  ["Long-Term Service Agreement - Bronze",
                              "Long-Term Service Agreement - Gold",
                              "Preventive Maintenance Contract",
                              "Emergency Response SLA"],
    }
    names = [random.choice(product_names[f]) + f" ({fake.bothify('##?').upper()})"
             for f in families]

    df = pd.DataFrame({
        "PRODUCT_ID":     PRODUCT_IDS,
        "PART_NUMBER":    [f"PN-{fake.bothify('####-???-##').upper()}" for _ in range(N)],
        "NAME":           names,
        "PRODUCT_FAMILY": families,
        "UNIT_COST_EUR":  np.round(np.random.exponential(scale=15000, size=N).clip(500, 250000), 2),
        "LEAD_TIME_DAYS": nullable(list(np.random.randint(1, 120, N)), 0.10),
        "STOCK_AVAILABLE":np.random.randint(0, 50, N),
        # null ~15% — not all products rated
        "COMPATIBILITY":  nullable(
            list(np.random.choice(EQUIPMENT_CLASSES, N)), 0.15
        ),
    })
    return df

# ── 7. CAMPAIGN_VW ────────────────────────────────────────────────
def gen_campaign_vw():
    N = 30
    campaign_names = [
        "Fleet-Wide Rotor Health Retrofit 2026",
        "Cybersecurity Firmware Patch for Gas Controls",
        "Offshore Wind Blade Inspection Program",
        "Decarbonisation Retrofit EMEA 2025",
        "Digital Twin Rollout — Gas Fleet",
        "APAC Transformer Overhaul Initiative",
        "Emergency Response Readiness 2024",
        "High Voltage System Modernisation",
        "Predictive Maintenance AI Deployment",
        "Grid Frequency Stability Program",
        "SGT-800 Life Extension Campaign",
        "Remote Operations Center Setup",
        "Thermal Efficiency Improvement Drive",
        "Blade Erosion Mitigation Program",
        "Control System Standardisation",
        "Americas Fleet Safety Audit 2025",
        "Renewable Integration Grid Study",
        "Compressor Wash Optimization",
        "Generator Rewind Initiative",
        "Substation Automation Upgrade",
        "Carbon Capture Readiness Study",
        "Fleet-Wide Oil Analysis Program",
        "Vibration Monitoring Rollout",
        "SCADA Cybersecurity Hardening",
        "Emergency Spare Parts Pre-positioning",
        "SWT Wind Fleet Performance Review",
        "Long-Term Service Contract Renewal",
        "Digital Controls Migration v2",
        "Cooling System Efficiency Audit",
        "Power Factor Correction Initiative",
    ]
    start_dates = rand_dates(N, start=datetime(2022,1,1), end=datetime(2024,6,30))
    end_dates   = [s + timedelta(days=random.randint(90, 730)) for s in start_dates]

    df = pd.DataFrame({
        "CAMPAIGN_ID":    CAMPAIGN_IDS,
        "CAMPAIGN_NAME":  campaign_names[:N],
        "OWNER_ID":       np.random.choice(USER_IDS, N),  # FK → USER_VW
        "OBJECTIVE":      np.random.choice(
            ["Efficiency Gain", "Safety Compliance", "Life Extension",
             "Cost Reduction", "Regulatory Compliance"],
            N, p=[0.25, 0.25, 0.20, 0.15, 0.15]
        ),
        "BUDGET_EUR":     np.round(np.random.uniform(50000, 5000000, N), 2),
        "STATUS":         np.random.choice(
            ["Active", "Completed", "Planning", "On Hold"],
            N, p=[0.35, 0.30, 0.25, 0.10]
        ),
        "START_DATE":     [s.date() for s in start_dates],
        "END_DATE":       [e.date() for e in end_dates],
        "REGION":         np.random.choice(["EMEA", "APAC", "Americas", "Global"],
                                            N, p=[0.35, 0.25, 0.25, 0.15]),
        # null ~10%
        "APPROVED_BY":    nullable(
            [f"DIR-{str(random.randint(1,10)).zfill(3)}" for _ in range(N)], 0.10
        ),
    })
    return df

# ── 8. EVENT_VW ───────────────────────────────────────────────────
def gen_event_vw():
    N = 500
    event_ids   = [f"EVT-{str(i).zfill(6)}" for i in range(1, N+1)]
    planned     = rand_dates(N, start=datetime(2022,1,1), end=datetime(2025,6,30))

    df = pd.DataFrame({
        "EVENT_ID":        event_ids,
        "CAMPAIGN_ID":     np.random.choice(CAMPAIGN_IDS, N),   # FK → CAMPAIGN_VW
        "PRODUCT_ID":      np.random.choice(PRODUCT_IDS, N),    # FK → PRODUCT2_VW
        "EVENT_TYPE":      np.random.choice(
            ["Site Audit", "Parts Delivery", "Software Deployment",
             "Engineering Milestone", "Training Session", "Inspection Visit"],
            N, p=[0.20, 0.25, 0.20, 0.15, 0.10, 0.10]
        ),
        "PLANNED_DATE":    [d.date() for d in planned],
        # Completed 50%, Scheduled 35%, Delayed 15%
        "EXECUTION_STATUS":np.random.choice(
            ["Completed", "Scheduled", "Delayed"],
            N, p=[0.50, 0.35, 0.15]
        ),
        # null ~20% — not all events have an assigned engineer yet
        "ASSIGNED_ENGINEER": nullable(
            list(np.random.choice(USER_IDS, N)), 0.20
        ),
        "ACTUAL_DATE":     nullable(
            [d.date() for d in rand_dates(N)], 0.35
        ),
        "COST_EUR":        nullable(
            list(np.round(np.random.exponential(scale=5000, size=N).clip(100, 50000), 2)),
            0.25
        ),
        "NOTES":           nullable(
            [fake.sentence(nb_words=8) for _ in range(N)], 0.30
        ),
    })
    return df


# ═══════════════════════════════════════════════════════════════════
# Load into SQLite
# ═══════════════════════════════════════════════════════════════════

def load_to_sqlite(db_path, tables: dict):
    conn = sqlite3.connect(db_path)
    for table_name, df in tables.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"  ✓ {table_name}: {len(df):,} rows → {db_path}")
    conn.close()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("═" * 55)
    print("Generating Domain 1 — Operational & Power Grid")
    print("═" * 55)
    asset_df    = gen_asset_vw()
    outage_df   = gen_outage_vw()
    case_df     = gen_case_vw(outage_df["OUTAGE_ID"].tolist())
    window_df   = gen_service_window_vw()

    load_to_sqlite("domain1.db", {
        "ASSET_VW":                asset_df,
        "OUTAGE__C_VW":            outage_df,
        "CASE_VW":                 case_df,
        "SERVICE_WINDOW_C_VW_TEST":window_df,
    })

    print("\n" + "═" * 55)
    print("Generating Domain 2 — Energy Service Campaigns")
    print("═" * 55)
    user_df     = gen_user_vw()
    product_df  = gen_product2_vw()
    campaign_df = gen_campaign_vw()
    event_df    = gen_event_vw()

    load_to_sqlite("domain2.db", {
        "USER_VW":      user_df,
        "PRODUCT2_VW":  product_df,
        "CAMPAIGN_VW":  campaign_df,
        "EVENT_VW":     event_df,
    })

    print("\n═" * 55)
    print("\n✅ Done. Summary:")
    print(f"{'Table':<30} {'Rows':>8}  DB")
    print("-" * 50)
    for name, df in [
        ("ASSET_VW",                asset_df),
        ("OUTAGE__C_VW",            outage_df),
        ("CASE_VW",                 case_df),
        ("SERVICE_WINDOW_C_VW_TEST",window_df),
    ]:
        print(f"  {name:<28} {len(df):>8,}  domain1.db")
    for name, df in [
        ("USER_VW",     user_df),
        ("PRODUCT2_VW", product_df),
        ("CAMPAIGN_VW", campaign_df),
        ("EVENT_VW",    event_df),
    ]:
        print(f"  {name:<28} {len(df):>8,}  domain2.db")

    print("\nCross-domain conceptual link:")
    print("  CASE_VW.ASSET_ID (domain1) ↔ ASSET_VW.ASSET_ID (domain1)")
    print("  CAMPAIGN_VW.OWNER_ID       ↔ USER_VW.USER_ID (domain2)")
    print("  EVENT_VW.CAMPAIGN_ID       ↔ CAMPAIGN_VW.CAMPAIGN_ID (domain2)")
    print("  EVENT_VW.PRODUCT_ID        ↔ PRODUCT2_VW.PRODUCT_ID (domain2)")
    print("  [cross-DB] CASE_VW.ASSET_ID concept shared with CAMPAIGN_VW.REGION")

if __name__ == "__main__":
    main()
