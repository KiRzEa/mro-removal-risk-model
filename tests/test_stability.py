import numpy as np
import pytest

from src.stability import summarize_runs


def test_summary_has_the_expected_statistics():
    summary = summarize_runs([0.6, 0.7, 0.8, 0.9, 1.0], target=0.8)
    assert summary["runs_used"] == 5
    assert summary["mean"] == pytest.approx(0.8)
    assert summary["median"] == pytest.approx(0.8)
    assert summary["std"] == pytest.approx(np.std([0.6, 0.7, 0.8, 0.9, 1.0], ddof=1))
    assert summary["p2_5"] == pytest.approx(np.percentile([0.6, 0.7, 0.8, 0.9, 1.0], 2.5))
    assert summary["p97_5"] == pytest.approx(np.percentile([0.6, 0.7, 0.8, 0.9, 1.0], 97.5))
    assert summary["share_at_or_above_target"] == pytest.approx(0.6)


def test_the_same_input_gives_the_same_summary():
    values = list(np.random.default_rng(0).random(25))
    assert summarize_runs(values, 0.8) == summarize_runs(values, 0.8)


def test_fewer_runs_than_required_is_an_error():
    with pytest.raises(ValueError, match="at least 20"):
        summarize_runs([0.8] * 5, target=0.8, minimum_runs=20)


def test_runs_without_a_value_are_dropped_and_counted():
    summary = summarize_runs([0.9, float("nan"), 0.7], target=0.8)
    assert summary["runs_used"] == 2
    assert summary["runs_requested"] == 3
    assert summary["share_at_or_above_target"] == pytest.approx(0.5)
