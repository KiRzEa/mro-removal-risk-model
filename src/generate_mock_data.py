import argparse
import copy
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.schema import COLUMN_NAMES, LABEL_COLUMN, SENSOR_NAMES, write_schema_markdown

CYCLES_PER_BLOCK = 30
BLOCK_COUNT = 25
START_DATE = pd.Timestamp("2022-01-01")
DAYS_PER_BLOCK = 30
DRIFT_START_BLOCK = 20
WEAR_OUT_SHAPE = 3.0
PRECURSOR_BLOCKS_RANGE = (8, 14)
SCHEDULED_REMOVAL_PROBABILITY = 0.004
SENSOR_DROPOUT_PROBABILITY = 0.004
SENSOR_SPIKE_PROBABILITY = 0.005
LIFE_SCALE = 1.25
MAX_CYCLES_SINCE_FAULT = 900
MAX_DAYS_SINCE_MAINTENANCE = 365
SHIFTED_WINDOW_SERIAL_START = 1_000_000

FAULT_CODES_LOW_RISK = ["F100", "F210", "F310"]
FAULT_CODES_HIGH_RISK = ["F415", "F520"]
OPERATORS = ["OP_A", "OP_B", "OP_C"]


@dataclass(frozen=True)
class Difficulty:
    sudden_failure_share: float
    precursor_strength: float
    sensor_noise_scale: float
    missing_rate: float
    label_drop_rate: float
    drift_strength: float


DIFFICULTIES = {
    "easy": Difficulty(0.05, 4.0, 0.8, 0.02, 0.0, 0.0),
    "realistic": Difficulty(0.15, 3.0, 1.0, 0.06, 0.015, 0.3),
    "hard": Difficulty(0.35, 2.0, 1.3, 0.15, 0.04, 0.8),
}


@dataclass(frozen=True)
class ComponentType:
    name: str
    mean_life_blocks: float
    sensor_baselines: tuple[float, float, float, float]
    overhaul_interval_blocks: int


COMPONENT_TYPES = [
    ComponentType("hydraulic_pump", 42, (2.4, 74.0, 305.0, 20.0), 16),
    ComponentType("fuel_pump", 46, (1.8, 68.0, 280.0, 18.0), 18),
    ComponentType("starter_generator", 40, (3.0, 88.0, 330.0, 26.0), 14),
    ComponentType("bleed_air_valve", 48, (1.5, 95.0, 260.0, 12.0), 20),
    ComponentType("brake_assembly", 44, (2.8, 82.0, 295.0, 16.0), 15),
]

AIRCRAFT_TYPES = {"A320": 1.9, "B737": 2.1, "ATR72": 1.1}
SENSOR_NOISE_FRACTION = (0.10, 0.03, 0.025, 0.06)
SENSOR_DIRECTION = (1.0, 1.0, -1.0, 1.0)


@dataclass(frozen=True)
class Shift:
    sensor_offset_sd: float = 0.0
    utilization_factor: float = 1.0
    extra_missing_rate: float = 0.0
    life_scale_factor: float = 1.0
    precursor_factor: float = 1.0
    early_failure_share: float = 0.0


@dataclass
class Serial:
    number: int
    created_block: int
    life_blocks: int
    age_blocks: int
    is_sudden: bool
    precursor_blocks: int
    sensor_offsets: np.ndarray
    informative_missing: bool


def draw_life_blocks(rng, component_type, part_number):
    mean_life = component_type.mean_life_blocks * LIFE_SCALE
    if part_number.endswith("B"):
        mean_life *= 0.85
    scale = mean_life / math.gamma(1 + 1 / WEAR_OUT_SHAPE)
    return max(3, int(round(scale * rng.weibull(WEAR_OUT_SHAPE))))


def new_serial(rng, component_type, part_number, number, difficulty, shift, block, start_age=None):
    life_blocks = draw_life_blocks(rng, component_type, part_number)
    age_blocks = int(rng.uniform(0, life_blocks - 1)) if start_age is None else start_age
    serial = Serial(
        number=number,
        created_block=block,
        life_blocks=life_blocks,
        age_blocks=age_blocks,
        is_sudden=bool(rng.random() < difficulty.sudden_failure_share),
        precursor_blocks=int(rng.integers(*PRECURSOR_BLOCKS_RANGE)),
        sensor_offsets=rng.normal(0, 0.5, len(SENSOR_NAMES)),
        informative_missing=bool(rng.random() < 0.2),
    )
    if block >= DRIFT_START_BLOCK:
        if shift.life_scale_factor != 1.0:
            serial.life_blocks = max(3, round(serial.life_blocks * shift.life_scale_factor))
        if shift.precursor_factor != 1.0:
            serial.precursor_blocks = scale_precursor(serial.precursor_blocks, shift)
        if shift.early_failure_share > 0 and rng.random() < shift.early_failure_share:
            serial.life_blocks = int(rng.integers(2, 6))
    return serial


def scale_precursor(precursor_blocks, shift):
    return max(2, round(precursor_blocks * shift.precursor_factor))


def apply_shift_to_active_serial(serial, shift):
    if shift.life_scale_factor != 1.0:
        remaining = serial.life_blocks - serial.age_blocks
        scaled_remaining = max(2, round(remaining * shift.life_scale_factor))
        serial.life_blocks = serial.age_blocks + scaled_remaining
    if shift.precursor_factor != 1.0:
        serial.precursor_blocks = scale_precursor(serial.precursor_blocks, shift)


def precursor_progress(serial, blocks_remaining):
    if serial.is_sudden:
        return 0.0
    progress = (serial.precursor_blocks - blocks_remaining + 1) / serial.precursor_blocks
    return float(np.clip(progress, 0.0, 1.0))


def simulate_slot(rng, difficulty, shift, aircraft, component_type, take_serial_number):
    part_number = f"{component_type.name[:3].upper()}-{rng.choice(['A', 'B'])}"
    serial = new_serial(
        rng, component_type, part_number, take_serial_number(0), difficulty, shift, 0
    )
    prior_removals = int(rng.poisson(0.6))
    cycles_since_fault = MAX_CYCLES_SINCE_FAULT
    last_fault_code = "NONE"
    days_since_maintenance = int(rng.integers(10, 200))
    dropout_blocks_left = [0] * len(SENSOR_NAMES)
    rows = []

    for block in range(BLOCK_COUNT):
        if block == DRIFT_START_BLOCK and serial.created_block < DRIFT_START_BLOCK:
            apply_shift_to_active_serial(serial, shift)
        blocks_remaining = serial.life_blocks - serial.age_blocks
        is_failing_block = blocks_remaining == 1
        progress = precursor_progress(serial, blocks_remaining)
        wear_fraction = serial.age_blocks / serial.life_blocks
        is_drifting = block >= DRIFT_START_BLOCK

        sensor_values = {}
        for index, name in enumerate(SENSOR_NAMES):
            baseline = component_type.sensor_baselines[index]
            noise_sd = baseline * SENSOR_NOISE_FRACTION[index] * difficulty.sensor_noise_scale
            degradation_step = SENSOR_DIRECTION[index] * noise_sd
            mean = baseline + serial.sensor_offsets[index] * noise_sd
            mean += degradation_step * difficulty.precursor_strength * progress**1.5
            mean += degradation_step * 0.5 * wear_fraction**2
            mean += rng.normal(0, noise_sd)
            if is_drifting and aircraft["operator"] == "OP_C" and name == "oil_temp_c":
                mean += noise_sd * (2.0 * difficulty.drift_strength + shift.sensor_offset_sd)
            if rng.random() < SENSOR_SPIKE_PROBABILITY:
                mean += rng.choice([-1, 1]) * rng.uniform(4, 6) * noise_sd
            spread = noise_sd * 0.8 * (1 + 1.5 * progress) * rng.uniform(0.9, 1.1)

            if dropout_blocks_left[index] == 0 and rng.random() < SENSOR_DROPOUT_PROBABILITY:
                dropout_blocks_left[index] = int(rng.integers(3, 7))
            in_dropout = dropout_blocks_left[index] > 0
            dropout_blocks_left[index] = max(0, dropout_blocks_left[index] - 1)
            sensor_is_failing = (
                serial.informative_missing
                and not serial.is_sudden
                and blocks_remaining <= 2
                and index == 0
            )
            missing_rate = difficulty.missing_rate + (
                shift.extra_missing_rate if is_drifting else 0
            )
            is_missing = in_dropout or sensor_is_failing or rng.random() < missing_rate
            sensor_values[f"{name}_mean"] = np.nan if is_missing else mean
            sensor_values[f"{name}_std"] = np.nan if is_missing else spread

        fault_count = int(rng.poisson(0.25 + 3.5 * progress))
        if fault_count > 0:
            high_risk_probability = 0.1 + 0.6 * progress
            pool = (
                FAULT_CODES_HIGH_RISK
                if rng.random() < high_risk_probability
                else FAULT_CODES_LOW_RISK
            )
            last_fault_code = str(rng.choice(pool))
            distinct_codes = int(min(fault_count, rng.integers(1, fault_count + 1)))
            cycles_since_fault = int(rng.integers(0, CYCLES_PER_BLOCK))
        else:
            distinct_codes = 0
            cycles_since_fault = min(cycles_since_fault + CYCLES_PER_BLOCK, MAX_CYCLES_SINCE_FAULT)

        maintenance_actions = int(rng.poisson(0.35 + 1.2 * progress))
        if maintenance_actions > 0:
            days_since_maintenance = int(rng.integers(0, DAYS_PER_BLOCK))
        else:
            days_since_maintenance = min(
                days_since_maintenance + DAYS_PER_BLOCK, MAX_DAYS_SINCE_MAINTENANCE
            )

        cycles_since_new = serial.age_blocks * CYCLES_PER_BLOCK + int(
            rng.integers(0, CYCLES_PER_BLOCK)
        )
        utilization = 1 + (0.25 * difficulty.drift_strength if is_drifting else 0.0)
        if is_drifting:
            utilization *= shift.utilization_factor
        hours_per_cycle = aircraft["hours_per_cycle"]

        rows.append(
            {
                "component_id": f"C{serial.number:05d}",
                "slot_id": f"{aircraft['aircraft_id']}-{component_type.name}",
                "aircraft_id": aircraft["aircraft_id"],
                "snapshot_date": START_DATE + pd.Timedelta(days=DAYS_PER_BLOCK * block),
                "component_type": component_type.name,
                "part_number": part_number,
                "aircraft_type": aircraft["aircraft_type"],
                "operator": aircraft["operator"],
                "aircraft_age_years": aircraft["base_age_years"] + block * DAYS_PER_BLOCK / 365,
                "flight_cycles_since_new": cycles_since_new,
                "flight_hours_since_new": cycles_since_new
                * hours_per_cycle
                * rng.normal(1.0, 0.02),
                "cycles_since_overhaul": (
                    serial.age_blocks % component_type.overhaul_interval_blocks
                )
                * CYCLES_PER_BLOCK
                + int(rng.integers(0, CYCLES_PER_BLOCK)),
                "flight_hours_last_30_cycles": CYCLES_PER_BLOCK
                * hours_per_cycle
                * rng.normal(1.0, 0.05)
                * utilization,
                **sensor_values,
                "fault_count_last_30_cycles": fault_count,
                "distinct_fault_codes_last_30_cycles": distinct_codes,
                "last_fault_code": last_fault_code,
                "cycles_since_last_fault": cycles_since_fault,
                "maintenance_actions_last_90_days": maintenance_actions,
                "days_since_last_maintenance": days_since_maintenance,
                "prior_unscheduled_removals_in_slot": prior_removals,
                LABEL_COLUMN: int(is_failing_block),
            }
        )

        if is_failing_block:
            prior_removals += 1
            serial = new_serial(
                rng,
                component_type,
                part_number,
                take_serial_number(block + 1),
                difficulty,
                shift,
                block + 1,
                0,
            )
            cycles_since_fault, last_fault_code = MAX_CYCLES_SINCE_FAULT, "NONE"
        elif rng.random() < SCHEDULED_REMOVAL_PROBABILITY:
            serial = new_serial(
                rng,
                component_type,
                part_number,
                take_serial_number(block + 1),
                difficulty,
                shift,
                block + 1,
                0,
            )
            cycles_since_fault, last_fault_code = MAX_CYCLES_SINCE_FAULT, "NONE"
        else:
            serial.age_blocks += 1

    return rows


def serial_number_taker(serial_numbers, window_serial_numbers=None):
    def take(block):
        if window_serial_numbers is not None and block >= DRIFT_START_BLOCK:
            return next(window_serial_numbers)
        return next(serial_numbers)

    return take


def generate(seed, difficulty_name, aircraft_count, shift: Shift | None = None):
    shift = shift or Shift()
    rng = np.random.default_rng(seed)
    difficulty = DIFFICULTIES[difficulty_name]
    window_serial_numbers = itertools.count(SHIFTED_WINDOW_SERIAL_START)
    is_stressed = shift != Shift()
    next_serial_number = 1
    rows = []
    for aircraft_index in range(aircraft_count):
        aircraft_type = str(rng.choice(list(AIRCRAFT_TYPES)))
        aircraft = {
            "aircraft_id": f"AC{aircraft_index + 1:03d}",
            "aircraft_type": aircraft_type,
            "operator": str(rng.choice(OPERATORS)),
            "base_age_years": float(rng.uniform(2, 18)),
            "hours_per_cycle": AIRCRAFT_TYPES[aircraft_type],
        }
        for component_type in COMPONENT_TYPES:
            slot_start = next_serial_number
            # Shifted slots run on a copy of the random stream while the real stream plays the
            # unshifted slot, so a shift never moves the random numbers of later slots.
            shifted_stream = copy.deepcopy(rng) if is_stressed else None
            slot_numbers = itertools.count(slot_start)
            slot_rows = simulate_slot(
                rng,
                difficulty,
                Shift(),
                aircraft,
                component_type,
                serial_number_taker(slot_numbers),
            )
            next_serial_number = next(slot_numbers)
            if is_stressed:
                take_serial_number = serial_number_taker(
                    itertools.count(slot_start), window_serial_numbers
                )
                slot_rows = simulate_slot(
                    shifted_stream, difficulty, shift, aircraft, component_type, take_serial_number
                )
            rows.extend(slot_rows)

    data = pd.DataFrame(rows)
    dropped_label = (data[LABEL_COLUMN] == 1) & (rng.random(len(data)) < difficulty.label_drop_rate)
    data.loc[dropped_label, LABEL_COLUMN] = 0
    data = data.sort_values(["snapshot_date", "aircraft_id", "slot_id"]).reset_index(drop=True)
    return data[COLUMN_NAMES]


def describe(data):
    sensor_columns = [column for column in data.columns if column.endswith("_mean")]
    return {
        "rows": len(data),
        "components": data["component_id"].nunique(),
        "positives": int(data[LABEL_COLUMN].sum()),
        "positive_rate": round(float(data[LABEL_COLUMN].mean()), 4),
        "sensor_missing_rate": round(float(data[sensor_columns].isna().mean().mean()), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate the mock component snapshot dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", choices=list(DIFFICULTIES), default="realistic")
    parser.add_argument("--aircraft", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("data/mock_components.csv"))
    parser.add_argument("--shift-sensor-offset", type=float, default=0.0)
    parser.add_argument("--shift-usage", type=float, default=1.0)
    parser.add_argument("--shift-missing", type=float, default=0.0)
    parser.add_argument("--shift-life-scale", type=float, default=1.0)
    parser.add_argument("--shift-precursor", type=float, default=1.0)
    parser.add_argument("--shift-early-failures", type=float, default=0.0)
    args = parser.parse_args()

    shift = Shift(
        sensor_offset_sd=args.shift_sensor_offset,
        utilization_factor=args.shift_usage,
        extra_missing_rate=args.shift_missing,
        life_scale_factor=args.shift_life_scale,
        precursor_factor=args.shift_precursor,
        early_failure_share=args.shift_early_failures,
    )
    data = generate(args.seed, args.difficulty, args.aircraft, shift)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    write_schema_markdown(args.output.parent / "SCHEMA.md")
    print(args.output, describe(data))


if __name__ == "__main__":
    main()
