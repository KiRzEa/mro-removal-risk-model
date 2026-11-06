from pathlib import Path

import pandas as pd
import pytest

from src.generate_mock_data import (
    BLOCK_COUNT,
    DAYS_PER_BLOCK,
    DIFFICULTIES,
    DRIFT_START_BLOCK,
    START_DATE,
    Shift,
    generate,
)
from src.schema import COLUMN_NAMES, LABEL_COLUMN, SENSOR_COLUMNS


@pytest.fixture(scope="module")
def realistic_data():
    return generate(seed=42, difficulty_name="realistic", aircraft_count=80)


def test_same_seed_gives_the_same_data():
    first = generate(seed=1, difficulty_name="realistic", aircraft_count=10)
    second = generate(seed=1, difficulty_name="realistic", aircraft_count=10)
    assert first.equals(second)


def test_columns_follow_the_schema(realistic_data):
    assert list(realistic_data.columns) == COLUMN_NAMES
    assert set(realistic_data[LABEL_COLUMN].unique()) <= {0, 1}


def test_about_two_percent_of_rows_are_positive(realistic_data):
    assert 0.015 <= realistic_data[LABEL_COLUMN].mean() <= 0.025


def test_row_count_is_in_the_expected_range(realistic_data):
    assert 9_000 <= len(realistic_data) <= 11_000


@pytest.mark.parametrize("difficulty_name", list(DIFFICULTIES))
def test_every_difficulty_keeps_the_positive_rate(difficulty_name):
    data = generate(seed=7, difficulty_name=difficulty_name, aircraft_count=80)
    assert 0.015 <= data[LABEL_COLUMN].mean() <= 0.025


def test_harder_data_has_more_missing_sensor_values():
    easy = generate(seed=3, difficulty_name="easy", aircraft_count=20)
    hard = generate(seed=3, difficulty_name="hard", aircraft_count=20)
    assert hard[SENSOR_COLUMNS].isna().mean().mean() > easy[SENSOR_COLUMNS].isna().mean().mean()


def test_a_replacement_component_gets_a_new_id(realistic_data):
    positive_slots = realistic_data.loc[realistic_data[LABEL_COLUMN] == 1, "slot_id"].unique()[:20]
    for slot_id in positive_slots:
        ids_in_slot = realistic_data.loc[realistic_data["slot_id"] == slot_id, "component_id"]
        assert ids_in_slot.nunique() > 1


DRIFT_START_DATE = START_DATE + pd.Timedelta(days=DAYS_PER_BLOCK * DRIFT_START_BLOCK)
SENSOR_MEAN_COLUMNS = [column for column in SENSOR_COLUMNS if column.endswith("_mean")]


def split_at_drift(data):
    in_window = data["snapshot_date"] >= DRIFT_START_DATE
    return data[~in_window], data[in_window]


@pytest.fixture(scope="module")
def baseline_80():
    return generate(seed=42, difficulty_name="realistic", aircraft_count=80)


def test_neutral_shift_leaves_the_data_unchanged():
    plain = generate(seed=5, difficulty_name="realistic", aircraft_count=10)
    neutral = generate(seed=5, difficulty_name="realistic", aircraft_count=10, shift=Shift())
    assert plain.equals(neutral)


def test_a_shift_changes_nothing_before_the_drift_window():
    stress = Shift(
        sensor_offset_sd=2.0,
        utilization_factor=1.3,
        extra_missing_rate=0.1,
        life_scale_factor=0.6,
        precursor_factor=0.5,
        early_failure_share=0.3,
    )
    plain = generate(seed=5, difficulty_name="realistic", aircraft_count=15)
    shifted = generate(seed=5, difficulty_name="realistic", aircraft_count=15, shift=stress)
    plain_before, plain_window = split_at_drift(plain)
    shifted_before, shifted_window = split_at_drift(shifted)
    assert plain_before.equals(shifted_before)
    assert not plain_window.equals(shifted_window)


def test_shorter_life_raises_the_positive_rate_in_the_window(baseline_80):
    shifted = generate(42, "realistic", 80, Shift(life_scale_factor=0.5))
    base_rate = split_at_drift(baseline_80)[1][LABEL_COLUMN].mean()
    shifted_rate = split_at_drift(shifted)[1][LABEL_COLUMN].mean()
    assert shifted_rate > 1.5 * base_rate


def test_early_failures_happen_within_five_snapshots_in_the_window_only(baseline_80):
    shifted = generate(42, "realistic", 80, Shift(early_failure_share=0.5))

    def early_failure_dates(data):
        snapshots = data.groupby("component_id")["snapshot_date"].agg(["min", "count"])
        failed = data.loc[data[LABEL_COLUMN] == 1, "component_id"].unique()
        started_new = snapshots["min"] > START_DATE
        return snapshots[snapshots.index.isin(failed) & started_new & (snapshots["count"] <= 5)]

    base_early = early_failure_dates(baseline_80)
    shifted_early = early_failure_dates(shifted)
    assert len(shifted_early) > 3 * max(len(base_early), 1)
    assert (shifted_early["min"] >= DRIFT_START_DATE).all()


def test_extra_missing_rate_raises_missingness_only_in_the_window():
    plain = generate(seed=9, difficulty_name="realistic", aircraft_count=40)
    shifted = generate(9, "realistic", 40, Shift(extra_missing_rate=0.2))
    plain_before, plain_window = split_at_drift(plain)
    shifted_before, shifted_window = split_at_drift(shifted)
    assert shifted_before[SENSOR_COLUMNS].isna().equals(plain_before[SENSOR_COLUMNS].isna())
    increase = (
        shifted_window[SENSOR_COLUMNS].isna().mean().mean()
        - plain_window[SENSOR_COLUMNS].isna().mean().mean()
    )
    assert increase == pytest.approx(0.2, abs=0.04)


def test_higher_utilization_raises_flight_hours_only_in_the_window():
    plain = generate(seed=9, difficulty_name="realistic", aircraft_count=40)
    shifted = generate(9, "realistic", 40, Shift(utilization_factor=1.5))
    column = "flight_hours_last_30_cycles"
    plain_before, plain_window = split_at_drift(plain)
    shifted_before, shifted_window = split_at_drift(shifted)
    assert shifted_before[column].equals(plain_before[column])
    assert shifted_window[column].mean() == pytest.approx(
        1.5 * plain_window[column].mean(), rel=0.1
    )


def test_every_block_is_present_for_each_slot(baseline_80):
    assert baseline_80.groupby("slot_id").size().eq(BLOCK_COUNT).all()


def test_committed_dataset_matches_the_regenerated_default(baseline_80):
    committed = Path(__file__).resolve().parents[1] / "data" / "mock_components.csv"
    assert baseline_80.to_csv(index=False) == committed.read_text()
