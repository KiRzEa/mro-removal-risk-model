import numpy as np
import pandas as pd
import pytest

from src.explain import (
    contributions,
    describe_feature,
    explain_rows,
    feature_label,
    top_reasons,
)
from src.features import FEATURE_COLUMNS, build_features
from src.generate_mock_data import generate
from src.schema import LABEL_COLUMN
from src.train import train_bundle


@pytest.fixture(scope="module")
def trained():
    snapshots = generate(seed=11, difficulty_name="easy", aircraft_count=40)
    return snapshots, train_bundle(snapshots, seed=11, tune=False)


def test_a_flagged_row_gets_at_least_three_plain_reasons():
    row = pd.Series(0.0, index=FEATURE_COLUMNS, dtype=object)
    row["vibration_mm_s_z_vs_type"] = 3.1
    row["fault_count_last_30_cycles"] = 4
    row["flight_cycles_since_new"] = 2400
    row["last_fault_code"] = "F415"
    weights = pd.Series(0.0, index=FEATURE_COLUMNS)
    weights["vibration_mm_s_z_vs_type"] = 1.2
    weights["fault_count_last_30_cycles"] = 0.8
    weights["flight_cycles_since_new"] = 0.5
    reasons = top_reasons(row, weights)
    assert len(reasons) == 3
    assert all(isinstance(reason, str) and len(reason) > 10 for reason in reasons)
    assert "ibration" in reasons[0]


def test_wording_follows_the_sign_of_the_value():
    assert "above" in describe_feature("vibration_mm_s_z_vs_type", 2.5)
    assert "below" in describe_feature("pressure_kpa_z_vs_type", -2.5)
    assert "rose" in describe_feature("oil_temp_c_z_change_vs_previous", 1.0)
    assert "fell" in describe_feature("oil_temp_c_z_change_vs_previous", -1.0)


def test_an_unknown_feature_falls_back_to_a_readable_name():
    assert "some new feature" in describe_feature("some_new_feature", 3.0).lower()
    assert feature_label("some_new_feature") == "some new feature"


def test_a_missing_value_is_described_as_not_reported():
    assert "not reported" in describe_feature("vibration_mm_s_z_vs_type", float("nan"))


def test_contributions_line_up_with_the_feature_columns(trained):
    snapshots, run = trained
    features = build_features(snapshots, run.bundle.baselines).head(30)
    values = contributions(run.bundle, features)
    assert list(values.columns) == FEATURE_COLUMNS
    assert values.index.equals(features.index)
    assert np.isfinite(values.to_numpy()).all()


def test_explain_rows_gives_three_reasons_for_each_alerted_row(trained):
    snapshots, run = trained
    rows = snapshots[snapshots[LABEL_COLUMN] == 1].index[:5]
    explained = explain_rows(run.bundle, snapshots, list(rows))
    assert set(explained) == set(rows)
    for reasons in explained.values():
        assert len(reasons) == 3
        assert all(reason.strip() for reason in reasons)
