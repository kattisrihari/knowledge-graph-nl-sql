"""
Siemens NL-to-SQL Demo — Data Generator
Outputs 5 CSVs ready for Snowflake COPY INTO
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

N = 100_000

# ── Shared ID pools (referential integrity across tables) ──────────────────────
MACHINE_IDS     = [f"MCH-{str(i).zfill(4)}" for i in range(1, 501)]       # 500 machines
TECHNICIAN_IDS  = [f"TECH-{str(i).zfill(4)}" for i in range(1, 201)]      # 200 technicians
SUPPLIER_IDS    = [f"SUP-{str(i).zfill(3)}" for i in range(1, 51)]        # 50 suppliers
PLANT_IDS       = [f"PLT-{str(i).zfill(2)}" for i in range(1, 11)]        # 10 plants

MACHINE_TYPES   = ["CNC Mill", "Lathe", "Robotic Arm", "Conveyor", "Press", "Welding Unit", "Inspection Bot"]
MACHINE_MODELS  = ["Siemens S7-1500", "SINUMERIK 840D", "SIMATIC ET200", "SIMOTION D", "SINAMICS G120", "SIRIUS 3RW"]
ISSUE_TYPES     = ["Overheating", "Vibration", "Electrical Fault", "Sensor Failure", "Lubrication", "Belt Wear", "Software Error"]
DEFECT_TYPES    = ["Surface Crack", "Dimensional Error", "Material Defect", "Assembly Fault", "Calibration Drift", "Weld Failure"]
PART_NAMES      = ["Drive Belt", "Servo Motor", "Control Board", "Bearing Assembly", "Hydraulic Seal",
                   "Proximity Sensor", "Power Supply Unit", "Encoder Disc", "Coolant Pump", "Safety Relay"]

START_DATE = datetime(2022, 1, 1)
END_DATE   = datetime(2024, 12, 31)

def rand_date(start=START_DATE, end=END_DATE):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

def rand_dates(n, start=START_DATE, end=END_DATE):
    span = int((end - start).total_seconds())
    offsets = np.random.randint(0, span, size=n)
    return [start + timedelta(seconds=int(o)) for o in offsets]


# ── 1. machines (operations schema) ───────────────────────────────────────────
print("Generating machines...")

install_dates = rand_dates(N, start=datetime(2015, 1, 1), end=datetime(2023, 12, 31))
last_service  = [d + timedelta(days=random.randint(30, 900)) for d in install_dates]
last_service  = [min(d, END_DATE) for d in last_service]

machines = pd.DataFrame({
    "machine_id":        np.random.choice(MACHINE_IDS, N),
    "plant_id":          np.random.choice(PLANT_IDS, N),
    "machine_type":      np.random.choice(MACHINE_TYPES, N, p=[0.2, 0.15, 0.2, 0.1, 0.1, 0.15, 0.1]),
    "model":             np.random.choice(MACHINE_MODELS, N),
    "install_date":      [d.date() for d in install_dates],
    "last_service_date": [d.date() for d in last_service],
    "status":            np.random.choice(["Active", "Idle", "Under Maintenance", "Decommissioned"],
                                          N, p=[0.65, 0.15, 0.15, 0.05]),
    "operational_hours": np.random.randint(500, 50000, N),
    "energy_consumption_kwh": np.round(np.random.uniform(5.0, 120.0, N), 2),
    "location_zone":     np.random.choice(["Zone-A", "Zone-B", "Zone-C", "Zone-D"], N),
})
# Deduplicate machine_id — keep first occurrence per machine, pad rest with new combos
machines.drop_duplicates(subset="machine_id", inplace=True)
# Reindex to ensure exactly N rows aren't needed for master — machines is a dimension
# Keep as-is; other tables reference from MACHINE_IDS pool
machines.to_csv(f"{OUT_DIR}/machines.csv", index=False)
print(f"  ✓ machines: {len(machines):,} rows")


# ── 2. maintenance_logs (operations schema) ────────────────────────────────────
print("Generating maintenance_logs...")

log_dates = rand_dates(N)
durations = np.round(np.random.exponential(scale=4.0, size=N).clip(0.5, 48.0), 1)
costs      = np.round(durations * np.random.uniform(80, 350, N), 2)

maintenance_logs = pd.DataFrame({
    "log_id":           [f"LOG-{str(i).zfill(7)}" for i in range(1, N + 1)],
    "machine_id":       np.random.choice(MACHINE_IDS, N),
    "technician_id":    np.random.choice(TECHNICIAN_IDS, N),
    "log_date":         [d.date() for d in log_dates],
    "issue_type":       np.random.choice(ISSUE_TYPES, N, p=[0.18, 0.14, 0.16, 0.15, 0.12, 0.13, 0.12]),
    "downtime_hrs":     durations,
    "repair_cost_usd":  costs,
    "resolved":         np.random.choice([True, False], N, p=[0.88, 0.12]),
    "priority":         np.random.choice(["Low", "Medium", "High", "Critical"], N, p=[0.3, 0.4, 0.2, 0.1]),
    "notes":            [fake.sentence(nb_words=8) for _ in range(N)],
})
maintenance_logs.to_csv(f"{OUT_DIR}/maintenance_logs.csv", index=False)
print(f"  ✓ maintenance_logs: {len(maintenance_logs):,} rows")


# ── 3. parts_inventory (operations schema) ─────────────────────────────────────
print("Generating parts_inventory...")

restock_dates = rand_dates(N, start=datetime(2021, 1, 1), end=END_DATE)

parts_inventory = pd.DataFrame({
    "part_id":            [f"PRT-{str(i).zfill(7)}" for i in range(1, N + 1)],
    "supplier_id":        np.random.choice(SUPPLIER_IDS, N),
    "machine_id":         np.random.choice(MACHINE_IDS, N),   # compatible machine
    "part_name":          np.random.choice(PART_NAMES, N),
    "stock_qty":          np.random.randint(0, 500, N),
    "unit_cost_usd":      np.round(np.random.uniform(10.0, 2500.0, N), 2),
    "last_restocked":     [d.date() for d in restock_dates],
    "reorder_threshold":  np.random.randint(10, 100, N),
    "warehouse_location": np.random.choice(["WH-01", "WH-02", "WH-03", "WH-04"], N),
    "lead_time_days":     np.random.randint(1, 45, N),
})
parts_inventory.to_csv(f"{OUT_DIR}/parts_inventory.csv", index=False)
print(f"  ✓ parts_inventory: {len(parts_inventory):,} rows")


# ── 4. production_batches (quality schema) ─────────────────────────────────────
print("Generating production_batches...")

batch_starts = rand_dates(N)
durations_h  = np.random.randint(1, 24, N)
batch_ends   = [s + timedelta(hours=int(d)) for s, d in zip(batch_starts, durations_h)]

units         = np.random.randint(50, 2000, N)
defect_rate   = np.round(np.random.beta(a=2, b=18, size=N), 4)   # realistic skew ~0–0.3
units_defect  = (units * defect_rate).astype(int)

production_batches = pd.DataFrame({
    "batch_id":          [f"BCH-{str(i).zfill(7)}" for i in range(1, N + 1)],
    "machine_id":        np.random.choice(MACHINE_IDS, N),        # FK → operations.machines
    "plant_id":          np.random.choice(PLANT_IDS, N),
    "start_time":        batch_starts,
    "end_time":          batch_ends,
    "units_produced":    units,
    "units_defective":   units_defect,
    "defect_rate":       defect_rate,
    "product_line":      np.random.choice(["PLC", "Drive", "Sensor", "Relay", "HMI"], N),
    "shift":             np.random.choice(["Morning", "Afternoon", "Night"], N),
    "batch_status":      np.random.choice(["Passed", "Failed", "Under Review"], N, p=[0.75, 0.15, 0.10]),
})
production_batches.to_csv(f"{OUT_DIR}/production_batches.csv", index=False)
print(f"  ✓ production_batches: {len(production_batches):,} rows")


# ── 5. defect_reports (quality schema) ────────────────────────────────────────
print("Generating defect_reports...")

# Pull batch_ids and part_ids from generated data for FK integrity
batch_ids = production_batches["batch_id"].tolist()
part_ids  = parts_inventory["part_id"].tolist()

detected_dates = rand_dates(N)

defect_reports = pd.DataFrame({
    "defect_id":       [f"DEF-{str(i).zfill(7)}" for i in range(1, N + 1)],
    "batch_id":        np.random.choice(batch_ids, N),            # FK → quality.production_batches
    "part_id":         np.random.choice(part_ids, N),             # FK → operations.parts_inventory
    "defect_type":     np.random.choice(DEFECT_TYPES, N, p=[0.2, 0.25, 0.15, 0.18, 0.12, 0.10]),
    "severity":        np.random.choice(["Minor", "Major", "Critical"], N, p=[0.55, 0.32, 0.13]),
    "detected_at":     detected_dates,
    "detection_method":np.random.choice(["Visual Inspection", "CMM", "X-Ray", "Ultrasonic", "Automated Vision"], N),
    "rework_required": np.random.choice([True, False], N, p=[0.45, 0.55]),
    "scrap_cost_usd":  np.round(np.random.exponential(scale=120.0, size=N).clip(0, 5000), 2),
    "root_cause":      np.random.choice(
        ["Supplier Material", "Machine Wear", "Operator Error", "Process Drift", "Environmental"],
        N, p=[0.25, 0.30, 0.15, 0.20, 0.10]
    ),
})
defect_reports.to_csv(f"{OUT_DIR}/defect_reports.csv", index=False)
print(f"  ✓ defect_reports: {len(defect_reports):,} rows")


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n✅ All CSVs written to ./data/")
print(f"{'Table':<25} {'Rows':>10} {'File'}")
print("-" * 55)
for fname, df in [
    ("operations.machines",          machines),
    ("operations.maintenance_logs",  maintenance_logs),
    ("operations.parts_inventory",   parts_inventory),
    ("quality.production_batches",   production_batches),
    ("quality.defect_reports",       defect_reports),
]:
    size_mb = os.path.getsize(f"{OUT_DIR}/{fname.split('.')[1]}.csv") / 1e6
    print(f"{fname:<25} {len(df):>10,}   ({size_mb:.1f} MB)")

print("\nCross-schema relationships established:")
print("  maintenance_logs.machine_id  → machines.machine_id")
print("  parts_inventory.machine_id   → machines.machine_id")
print("  production_batches.machine_id→ machines.machine_id")
print("  defect_reports.batch_id      → production_batches.batch_id")
print("  defect_reports.part_id       → parts_inventory.part_id")