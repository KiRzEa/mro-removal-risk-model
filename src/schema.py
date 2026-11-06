import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    unit: str
    description: str
    can_be_missing: bool = False


IDENTIFIER_COLUMNS = ["component_id", "slot_id", "aircraft_id"]
TIME_COLUMN = "snapshot_date"
LABEL_COLUMN = "removal_within_30_cycles"
CATEGORICAL_COLUMNS = [
    "component_type",
    "part_number",
    "aircraft_type",
    "operator",
    "last_fault_code",
]

SENSOR_NAMES = ["vibration_mm_s", "oil_temp_c", "pressure_kpa", "current_a"]
SENSOR_COLUMNS = [f"{name}_{statistic}" for name in SENSOR_NAMES for statistic in ("mean", "std")]

COLUMNS = [
    Column(
        "component_id",
        "string",
        "",
        "Serial of the installed component; a replacement gets a new id",
    ),
    Column("slot_id", "string", "", "Position on the aircraft (aircraft and component type)"),
    Column("aircraft_id", "string", "", "Aircraft tail identifier"),
    Column(
        "snapshot_date",
        "date",
        "",
        "Date the snapshot is taken; features use data before this date",
    ),
    Column(
        "component_type",
        "category",
        "",
        "hydraulic_pump, fuel_pump, starter_generator, bleed_air_valve, brake_assembly",
    ),
    Column("part_number", "category", "", "Part number variant of the component"),
    Column("aircraft_type", "category", "", "A320, B737 or ATR72"),
    Column("operator", "category", "", "Operating airline: OP_A, OP_B or OP_C"),
    Column("aircraft_age_years", "float", "years", "Age of the aircraft at the snapshot"),
    Column(
        "flight_cycles_since_new",
        "int",
        "cycles",
        "Cycles flown by this component since it was new",
    ),
    Column(
        "flight_hours_since_new",
        "float",
        "hours",
        "Flight hours of this component since it was new",
    ),
    Column("cycles_since_overhaul", "int", "cycles", "Cycles since the last scheduled overhaul"),
    Column(
        "flight_hours_last_30_cycles",
        "float",
        "hours",
        "Flight hours in the last 30 cycles (utilization)",
    ),
    Column("vibration_mm_s_mean", "float", "mm/s", "Mean vibration over the last 30 cycles", True),
    Column(
        "vibration_mm_s_std",
        "float",
        "mm/s",
        "Standard deviation of vibration over the last 30 cycles",
        True,
    ),
    Column(
        "oil_temp_c_mean", "float", "deg C", "Mean oil temperature over the last 30 cycles", True
    ),
    Column("oil_temp_c_std", "float", "deg C", "Standard deviation of oil temperature", True),
    Column("pressure_kpa_mean", "float", "kPa", "Mean pressure over the last 30 cycles", True),
    Column("pressure_kpa_std", "float", "kPa", "Standard deviation of pressure", True),
    Column("current_a_mean", "float", "A", "Mean current over the last 30 cycles", True),
    Column("current_a_std", "float", "A", "Standard deviation of current", True),
    Column("fault_count_last_30_cycles", "int", "count", "Fault messages in the last 30 cycles"),
    Column(
        "distinct_fault_codes_last_30_cycles",
        "int",
        "count",
        "Different fault codes in the last 30 cycles",
    ),
    Column("last_fault_code", "category", "", "Most recent fault code, or NONE"),
    Column(
        "cycles_since_last_fault",
        "int",
        "cycles",
        "Cycles since the last fault message, capped at 900",
    ),
    Column(
        "maintenance_actions_last_90_days",
        "int",
        "count",
        "Maintenance actions logged in the last 90 days",
    ),
    Column(
        "days_since_last_maintenance",
        "int",
        "days",
        "Days since the last maintenance action, capped at 365",
    ),
    Column(
        "prior_unscheduled_removals_in_slot",
        "int",
        "count",
        "Earlier unscheduled removals of components in this slot",
    ),
    Column(
        LABEL_COLUMN,
        "int",
        "0/1",
        "1 if the component is removed without a plan within the next 30 cycles",
    ),
]

COLUMN_NAMES = [column.name for column in COLUMNS]


def write_schema_markdown(path: Path) -> None:
    lines = [
        "# Mock dataset schema",
        "",
        "One row is one snapshot of an installed component every 30 flight cycles. "
        "Features describe the past only; the label looks at the next 30 cycles.",
        "",
        "| Column | Type | Unit | Can be missing | Description |",
        "|---|---|---|---|---|",
    ]
    for column in COLUMNS:
        missing = "yes" if column.can_be_missing else "no"
        lines.append(
            f"| {column.name} | {column.dtype} | {column.unit} | {missing} | {column.description} |"
        )
    path.write_text("\n".join(lines) + "\n")


REQUIRED_COLUMNS = [name for name in COLUMN_NAMES if name != LABEL_COLUMN]
NUMERIC_COLUMNS = [
    column.name
    for column in COLUMNS
    if column.dtype in ("int", "float") and column.name != LABEL_COLUMN
]


def validate_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if frame.empty:
        raise ValueError("The file has no rows")

    frame = frame.copy()
    try:
        frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN])
    except (ValueError, TypeError) as error:
        raise ValueError(f"{TIME_COLUMN} cannot be read as a date") from error
    if frame[TIME_COLUMN].isna().any():
        raise ValueError(f"{TIME_COLUMN} has empty or unreadable dates")

    for name in NUMERIC_COLUMNS + ([LABEL_COLUMN] if LABEL_COLUMN in frame.columns else []):
        converted = pd.to_numeric(frame[name], errors="coerce")
        if (converted.isna() & frame[name].notna()).any():
            raise ValueError(f"Column {name} has values that are not numbers")
        frame[name] = converted

    duplicates = int(frame.duplicated(["component_id", TIME_COLUMN]).sum())
    if duplicates:
        raise ValueError(f"{duplicates} rows repeat a (component_id, {TIME_COLUMN}) pair")

    extra_columns = [name for name in frame.columns if name not in COLUMN_NAMES]
    if extra_columns:
        warnings.warn(f"Ignoring extra columns: {', '.join(extra_columns)}", stacklevel=2)
        frame = frame.drop(columns=extra_columns)
    return frame


def load_snapshots(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise ValueError("The file is empty") from error
    return validate_snapshots(frame)
