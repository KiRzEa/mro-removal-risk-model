import json

import numpy as np
import pytest

from src.generate_mock_data import generate
from src.models import score_snapshots
from src.schema import LABEL_COLUMN, SENSOR_NAMES, TIME_COLUMN
from src.train import train_bundle


@pytest.fixture(scope="module")
def trained():
    snapshots = generate(seed=21, difficulty_name="easy", aircraft_count=40)
    return snapshots, train_bundle(snapshots, seed=21, tune=False).bundle


def a_component_with_long_history(snapshots):
    counts = snapshots.groupby("component_id").size()
    component = counts[counts >= 20].index[0]
    return snapshots[snapshots.component_id == component]


def test_normal_history_gets_normal_confidence(trained):
    snapshots, bundle = trained
    component = a_component_with_long_history(snapshots)
    scored = score_snapshots(bundle, component)
    assert (scored.confidence.iloc[3:] == "normal").sum() > 5
    assert scored.confidence.iloc[0] == "low"


def test_an_unseen_component_type_is_scored_with_low_confidence(trained):
    snapshots, bundle = trained
    rows = a_component_with_long_history(snapshots).copy()
    rows["component_type"] = "brand_new_type"
    scored = score_snapshots(bundle, rows)
    assert scored.risk_score.notna().all()
    assert (scored.confidence == "low").all()


def test_a_component_with_one_snapshot_is_scored_with_low_confidence(trained):
    snapshots, bundle = trained
    single = snapshots.head(1).copy()
    single["component_id"] = "C_NEW"
    scored = score_snapshots(bundle, single)
    assert len(scored) == 1 and np.isfinite(scored.risk_score.iloc[0])
    assert scored.confidence.iloc[0] == "low"


def test_a_row_with_every_sensor_missing_is_scored_with_low_confidence(trained):
    snapshots, bundle = trained
    rows = a_component_with_long_history(snapshots).copy()
    for name in SENSOR_NAMES:
        rows[f"{name}_mean"] = np.nan
        rows[f"{name}_std"] = np.nan
    scored = score_snapshots(bundle, rows)
    assert scored.risk_score.notna().all()
    assert (scored.confidence == "low").all()


def test_scoring_does_not_need_the_label(trained):
    snapshots, bundle = trained
    component = a_component_with_long_history(snapshots)
    with_label = score_snapshots(bundle, component)
    without_label = score_snapshots(bundle, component.drop(columns=[LABEL_COLUMN]))
    assert with_label.risk_score.equals(without_label.risk_score)


def test_scores_depend_only_on_rows_up_to_the_period(trained):
    snapshots, bundle = trained
    component = a_component_with_long_history(snapshots).sort_values(TIME_COLUMN)
    full = score_snapshots(bundle, component).risk_score
    truncated = score_snapshots(bundle, component.head(10)).risk_score
    assert np.allclose(full.iloc[:10].to_numpy(), truncated.to_numpy())


import math  # noqa: E402

import pandas as pd  # noqa: E402

from src.score import OUTPUT_COLUMNS, main  # noqa: E402


@pytest.fixture(scope="module")
def model_file(trained, tmp_path_factory):
    _, bundle = trained
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    bundle.save(path)
    return path


def write_input(frame, directory, name="input.csv"):
    path = directory / name
    frame.to_csv(path, index=False)
    return path


def run(model_file, input_path, output_path, *extra):
    return main(
        [
            "--input",
            str(input_path),
            "--model",
            str(model_file),
            "--output",
            str(output_path),
            *extra,
        ]
    )


@pytest.fixture(scope="module")
def small_file_frame():
    return generate(seed=31, difficulty_name="realistic", aircraft_count=12)


def test_missing_required_columns_stop_with_a_message_naming_them(
    model_file, small_file_frame, tmp_path, capsys
):
    frame = small_file_frame.drop(columns=["vibration_mm_s_mean", "operator"])
    code = run(model_file, write_input(frame, tmp_path), tmp_path / "out.csv")
    assert code != 0
    message = capsys.readouterr().err
    assert "vibration_mm_s_mean" in message and "operator" in message
    assert not (tmp_path / "out.csv").exists()


def test_duplicate_rows_stop_the_command(model_file, small_file_frame, tmp_path, capsys):
    frame = pd.concat([small_file_frame, small_file_frame.head(1)])
    assert run(model_file, write_input(frame, tmp_path), tmp_path / "out.csv") != 0
    assert "repeat" in capsys.readouterr().err


def test_an_unreadable_date_stops_the_command(model_file, small_file_frame, tmp_path, capsys):
    frame = small_file_frame.copy()
    frame["snapshot_date"] = frame["snapshot_date"].astype(str)
    frame.loc[frame.index[0], "snapshot_date"] = "not a date"
    assert run(model_file, write_input(frame, tmp_path), tmp_path / "out.csv") != 0
    assert "date" in capsys.readouterr().err


def test_text_in_a_numeric_column_stops_the_command(model_file, small_file_frame, tmp_path, capsys):
    frame = small_file_frame.copy().astype({"current_a_mean": object})
    frame.loc[frame.index[0], "current_a_mean"] = "high"
    assert run(model_file, write_input(frame, tmp_path), tmp_path / "out.csv") != 0
    assert "current_a_mean" in capsys.readouterr().err


def test_an_empty_file_stops_the_command(model_file, tmp_path, capsys):
    path = tmp_path / "empty.csv"
    path.write_text("")
    assert run(model_file, path, tmp_path / "out.csv") != 0
    assert "empty" in capsys.readouterr().err.lower()


def test_extra_columns_only_warn_and_the_label_is_optional(
    model_file, small_file_frame, tmp_path, capsys
):
    frame = small_file_frame.drop(columns=[LABEL_COLUMN]).assign(notes="hello")
    code = run(model_file, write_input(frame, tmp_path), tmp_path / "out.csv")
    assert code == 0
    assert "notes" in capsys.readouterr().err
    assert (tmp_path / "out.csv").exists()


def test_output_follows_the_input_order_and_the_contract(model_file, small_file_frame, tmp_path):
    shuffled = small_file_frame.sample(frac=1.0, random_state=0)
    assert run(model_file, write_input(shuffled, tmp_path), tmp_path / "out.csv") == 0
    output = pd.read_csv(tmp_path / "out.csv", parse_dates=[TIME_COLUMN])
    assert list(output.columns) == OUTPUT_COLUMNS
    assert len(output) == len(shuffled)
    assert list(output.component_id) == list(shuffled.component_id)
    assert list(output[TIME_COLUMN]) == list(shuffled[TIME_COLUMN])


def test_a_missing_input_file_stops_the_command_with_a_message(model_file, tmp_path, capsys):
    assert run(model_file, tmp_path / "does_not_exist.csv", tmp_path / "out.csv") == 2
    assert "error" in capsys.readouterr().err.lower()


def test_alerts_never_exceed_five_percent_of_a_period(model_file, small_file_frame, tmp_path):
    assert run(model_file, write_input(small_file_frame, tmp_path), tmp_path / "out.csv") == 0
    output = pd.read_csv(tmp_path / "out.csv")
    per_period = output.groupby(TIME_COLUMN).alert.agg(["sum", "size"])
    assert (per_period["sum"] <= per_period["size"].map(lambda size: math.ceil(0.05 * size))).all()
    assert (output.rank_in_period >= 1).all()


def test_repeated_runs_give_identical_output(model_file, small_file_frame, tmp_path):
    path = write_input(small_file_frame, tmp_path)
    run(model_file, path, tmp_path / "first.csv")
    run(model_file, path, tmp_path / "second.csv")
    assert (tmp_path / "first.csv").read_text() == (tmp_path / "second.csv").read_text()


def test_reasons_are_filled_for_alerts_and_for_every_row_on_request(
    model_file, small_file_frame, tmp_path
):
    path = write_input(small_file_frame, tmp_path)
    run(model_file, path, tmp_path / "alerts.csv")
    run(model_file, path, tmp_path / "all.csv", "--explain-all")
    alerts_only = pd.read_csv(tmp_path / "alerts.csv").fillna("")
    everything = pd.read_csv(tmp_path / "all.csv").fillna("")
    flagged = alerts_only.alert == 1
    assert (alerts_only.loc[flagged, "reason_1":"reason_3"] != "").all().all()
    assert (alerts_only.loc[~flagged, "reason_1":"reason_3"] == "").all().all()
    assert (everything.loc[:, "reason_1":"reason_3"] != "").all().all()


def test_end_to_end_on_a_file_from_another_seed(model_file, tmp_path):
    fresh = generate(seed=123, difficulty_name="realistic", aircraft_count=15).drop(
        columns=[LABEL_COLUMN]
    )
    fresh.loc[fresh.component_id == fresh.component_id.iloc[0], "operator"] = "OP_NEW"
    assert run(model_file, write_input(fresh, tmp_path), tmp_path / "out.csv") == 0
    output = pd.read_csv(tmp_path / "out.csv")
    assert len(output) == len(fresh)
    assert output.risk_score.between(0, 1).all()
    assert set(output.confidence) <= {"low", "normal"}


def test_a_labelled_file_is_graded_and_an_unlabelled_one_is_not(
    model_file, small_file_frame, tmp_path, capsys
):
    labelled = write_input(small_file_frame, tmp_path)
    metrics_path = tmp_path / "metrics.json"
    assert (
        run(model_file, labelled, tmp_path / "out.csv", "--metrics-output", str(metrics_path)) == 0
    )
    assert "grading on" in capsys.readouterr().out
    grade = json.loads(metrics_path.read_text())
    assert grade["removals"] == int(small_file_frame[LABEL_COLUMN].sum())
    assert 0 <= grade["recall_at_alert_budget"] <= 1
    assert grade["alerts_per_100"] <= 5.5

    unlabelled = write_input(small_file_frame.drop(columns=[LABEL_COLUMN]), tmp_path, "plain.csv")
    assert run(model_file, unlabelled, tmp_path / "out2.csv") == 0
    assert "grading on" not in capsys.readouterr().out
