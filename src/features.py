from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.schema import SENSOR_NAMES, TIME_COLUMN

HIGH_RISK_FAULT_CODES = ["F415", "F520"]

CATEGORICAL_FEATURES = [
    "component_type",
    "part_number",
    "aircraft_type",
    "operator",
    "last_fault_code",
]
COPIED_NUMERIC_FEATURES = [
    "flight_cycles_since_new",
    "flight_hours_since_new",
    "cycles_since_overhaul",
    "flight_hours_last_30_cycles",
    "fault_count_last_30_cycles",
    "distinct_fault_codes_last_30_cycles",
    "cycles_since_last_fault",
    "maintenance_actions_last_90_days",
    "days_since_last_maintenance",
    "prior_unscheduled_removals_in_slot",
]
SENSOR_VALUE_FEATURES = [
    f"{name}_{statistic}" for name in SENSOR_NAMES for statistic in ("mean", "std")
]
DEVIATION_SUFFIXES = [
    "z_vs_type",
    "spread_z_vs_type",
    "z_change_vs_previous",
    "z_slope_last_3",
    "z_vs_own_prior_mean",
]
DEVIATION_FEATURES = [f"{name}_{suffix}" for name in SENSOR_NAMES for suffix in DEVIATION_SUFFIXES]
OTHER_DERIVED_FEATURES = [f"{name}_is_missing" for name in SENSOR_NAMES] + [
    "fault_count_prior_3",
    "high_risk_fault_recent",
    "history_length",
]
NUMERIC_FEATURES = (
    COPIED_NUMERIC_FEATURES + SENSOR_VALUE_FEATURES + DEVIATION_FEATURES + OTHER_DERIVED_FEATURES
)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class FleetBaselines:
    table: pd.DataFrame


def fit_fleet_baselines(train_snapshots: pd.DataFrame) -> FleetBaselines:
    columns = [f"{name}_{statistic}" for name in SENSOR_NAMES for statistic in ("mean", "std")]
    return FleetBaselines(train_snapshots.groupby("component_type")[columns].agg(["mean", "std"]))


def _deviation(values: pd.Series, stats: pd.DataFrame, column: str) -> np.ndarray:
    spread = stats[(column, "std")].to_numpy()
    spread = np.where(spread > 0, spread, np.nan)
    return (values.to_numpy() - stats[(column, "mean")].to_numpy()) / spread


def build_features(snapshots: pd.DataFrame, baselines: FleetBaselines) -> pd.DataFrame:
    ordered = snapshots.sort_values(["component_id", TIME_COLUMN], kind="mergesort")
    component_ids = ordered["component_id"]
    # Unseen component types reindex to NaN rows, so their deviations stay missing.
    stats = baselines.table.reindex(ordered["component_type"].to_numpy())

    features = ordered[
        COPIED_NUMERIC_FEATURES + SENSOR_VALUE_FEATURES + CATEGORICAL_FEATURES
    ].copy()
    for name in SENSOR_NAMES:
        mean_column, spread_column = f"{name}_mean", f"{name}_std"
        deviation = pd.Series(
            _deviation(ordered[mean_column], stats, mean_column), index=ordered.index
        )
        spread_deviation = _deviation(ordered[spread_column], stats, spread_column)

        # History is read with shift, so a row never sees anything dated after it.
        by_component = deviation.groupby(component_ids)
        previous, two_back, three_back = (by_component.shift(lag) for lag in (1, 2, 3))
        own_prior_mean = pd.concat([previous, two_back, three_back], axis=1).mean(axis=1)

        features[f"{name}_z_vs_type"] = deviation
        features[f"{name}_spread_z_vs_type"] = spread_deviation
        features[f"{name}_z_change_vs_previous"] = deviation - previous
        features[f"{name}_z_slope_last_3"] = np.where(
            two_back.notna(), (deviation - two_back) / 2, deviation - previous
        )
        features[f"{name}_z_vs_own_prior_mean"] = deviation - own_prior_mean
        features[f"{name}_is_missing"] = ordered[mean_column].isna().astype(int)

    fault_counts = ordered["fault_count_last_30_cycles"].groupby(component_ids)
    prior_counts = pd.concat([fault_counts.shift(lag) for lag in (1, 2, 3)], axis=1)
    features["fault_count_prior_3"] = prior_counts.sum(axis=1, min_count=1)

    high_risk = ordered["last_fault_code"].isin(HIGH_RISK_FAULT_CODES).astype(float)
    high_risk_by_component = high_risk.groupby(component_ids)
    recent = pd.concat(
        [high_risk, high_risk_by_component.shift(1), high_risk_by_component.shift(2)], axis=1
    )
    features["high_risk_fault_recent"] = recent.max(axis=1)
    features["history_length"] = ordered.groupby("component_id", sort=False).cumcount()
    return features[FEATURE_COLUMNS].reindex(snapshots.index)
