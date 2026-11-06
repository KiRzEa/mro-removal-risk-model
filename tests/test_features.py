import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, build_features, fit_fleet_baselines
from src.generate_mock_data import generate
from src.schema import IDENTIFIER_COLUMNS, LABEL_COLUMN, TIME_COLUMN

HISTORY_FEATURE_MARKERS = (
    "change_vs_previous",
    "slope_last_3",
    "own_prior_mean",
    "fault_count_prior_3",
)


@pytest.fixture(scope="module")
def snapshots():
    return generate(seed=3, difficulty_name="realistic", aircraft_count=15)


@pytest.fixture(scope="module")
def baselines(snapshots):
    dates = np.sort(snapshots[TIME_COLUMN].unique())
    return fit_fleet_baselines(snapshots[snapshots[TIME_COLUMN] < dates[10]])


def test_features_at_a_period_do_not_depend_on_later_rows(snapshots, baselines):
    cutoff = np.sort(snapshots[TIME_COLUMN].unique())[12]
    later = snapshots[TIME_COLUMN] > cutoff

    altered = snapshots.copy()
    rng = np.random.default_rng(0)
    for column in altered.select_dtypes("number").columns:
        altered[column] = altered[column].astype(float)
        altered.loc[later, column] = rng.normal(size=int(later.sum())) * 1000
    altered.loc[later, "last_fault_code"] = "F415"

    original_features = build_features(snapshots, baselines)
    altered_features = build_features(altered, baselines)
    pd.testing.assert_frame_equal(
        original_features[~later], altered_features[~later], check_dtype=False
    )


def test_first_snapshot_of_a_component_has_no_history(snapshots, baselines):
    features = build_features(snapshots, baselines)
    first = features[features["history_length"] == 0]
    assert len(first) > 0
    for column in features.columns:
        if any(marker in column for marker in HISTORY_FEATURE_MARKERS):
            assert first[column].isna().all(), column


def test_type_deviation_uses_the_baselines_it_is_given(snapshots, baselines):
    rows = snapshots[snapshots["component_type"] == "hydraulic_pump"].head(5)
    shifted = baselines.table.copy()
    shifted.loc["hydraulic_pump", ("vibration_mm_s_mean", "mean")] = 100.0
    shifted.loc["hydraulic_pump", ("vibration_mm_s_mean", "std")] = 2.0
    features = build_features(rows, type(baselines)(shifted))
    expected = (rows["vibration_mm_s_mean"] - 100.0) / 2.0
    pd.testing.assert_series_equal(
        features["vibration_mm_s_z_vs_type"], expected, check_names=False
    )


def test_unseen_component_type_gives_missing_deviation_not_an_error(snapshots, baselines):
    rows = snapshots.head(20).copy()
    rows["component_type"] = "brand_new_type"
    features = build_features(rows, baselines)
    assert features["vibration_mm_s_z_vs_type"].isna().all()


def test_feature_list_has_no_identifier_date_label_or_calendar_trend():
    forbidden = set(IDENTIFIER_COLUMNS) | {TIME_COLUMN, LABEL_COLUMN, "aircraft_age_years"}
    assert not forbidden & set(FEATURE_COLUMNS)


def test_missing_sensor_gets_an_indicator(snapshots, baselines):
    features = build_features(snapshots, baselines)
    missing = snapshots["vibration_mm_s_mean"].isna()
    assert missing.any()
    assert (features.loc[missing, "vibration_mm_s_is_missing"] == 1).all()
    assert (features.loc[~missing, "vibration_mm_s_is_missing"] == 0).all()
