import numpy as np
import pandas as pd
import pytest

from src.evaluation import alert_count, flag_alerts, rank_within_period


def make_periods(sizes):
    return np.concatenate([np.full(size, index) for index, size in enumerate(sizes)])


def test_alert_count_is_rounded_up_without_float_noise():
    assert alert_count(400, 0.05) == 20
    assert alert_count(41, 0.05) == 3
    assert alert_count(1, 0.05) == 1
    assert alert_count(0, 0.05) == 0


def test_alerts_are_the_top_share_of_each_period_and_never_more():
    rng = np.random.default_rng(0)
    periods = make_periods([400, 400, 41])
    scores = rng.random(len(periods))
    alerts = flag_alerts(scores, periods, 0.05)
    for period, expected in zip([0, 1, 2], [20, 20, 3], strict=True):
        assert alerts[periods == period].sum() == expected


def test_rank_one_is_the_highest_score_in_its_period():
    periods = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.2, 0.9, 0.5, 0.1, 0.3])
    assert list(rank_within_period(scores, periods)) == [3, 1, 2, 2, 1]


def test_ties_break_by_position_so_results_repeat():
    periods = np.zeros(10, dtype=int)
    scores = np.full(10, 0.5)
    first = flag_alerts(scores, periods, 0.2)
    second = flag_alerts(scores, periods, 0.2)
    assert list(first) == list(second)
    assert list(np.flatnonzero(first)) == [0, 1]


def test_dates_work_as_periods():
    periods = pd.to_datetime(["2022-01-01"] * 4 + ["2022-01-31"] * 4).to_numpy()
    scores = np.arange(8, dtype=float)
    alerts = flag_alerts(scores, periods, 0.25)
    assert list(np.flatnonzero(alerts)) == [3, 7]


def test_empty_input_gives_empty_output():
    assert flag_alerts(np.array([]), np.array([]), 0.05).size == 0


def test_a_budget_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError, match="budget"):
        flag_alerts(np.array([0.1]), np.array([0]), 1.5)


from src.evaluation import (  # noqa: E402
    budget_needed_for_target,
    cluster_bootstrap_interval,
    fixed_cutoff_policy,
    precision_at_budget,
    recall_at_budget,
    recall_vs_alerts_curve,
    reliability_bins,
    target_check,
)


@pytest.fixture
def toy():
    periods = np.zeros(100, dtype=int)
    labels = np.zeros(100, dtype=int)
    labels[[0, 1, 2, 3]] = 1
    scores = np.linspace(1.0, 0.0, 100)
    scores[3] = 0.001
    return labels, scores, periods


def test_recall_and_precision_on_a_toy_period(toy):
    labels, scores, periods = toy
    assert recall_at_budget(labels, scores, periods, 0.05) == pytest.approx(0.75)
    assert precision_at_budget(labels, scores, periods, 0.05) == pytest.approx(3 / 5)


def test_cluster_bootstrap_interval_brackets_the_estimate_and_repeats(toy):
    labels, scores, periods = toy
    alerts = flag_alerts(scores, periods, 0.05)
    components = np.arange(100) // 2
    first = cluster_bootstrap_interval(labels, alerts, components, resamples=300, seed=1)
    second = cluster_bootstrap_interval(labels, alerts, components, resamples=300, seed=1)
    assert first == second
    estimate, low, high = first["recall"]
    assert low <= estimate <= high
    assert estimate == pytest.approx(0.75)


def test_recall_never_falls_as_the_budget_grows(toy):
    labels, scores, periods = toy
    curve = recall_vs_alerts_curve(labels, scores, periods, budgets=range(1, 11))
    assert list(curve["budget_per_100"]) == list(range(1, 11))
    assert curve["recall"].is_monotonic_increasing or (curve["recall"].diff().dropna() >= 0).all()


def test_budget_needed_for_target_is_none_when_unreachable(toy):
    labels, scores, periods = toy
    assert budget_needed_for_target(labels, scores, periods, target=0.7, max_share=0.10) == 0.03
    assert budget_needed_for_target(labels, scores, periods, target=1.0, max_share=0.10) is None


def test_fixed_cutoff_policy_reports_its_own_alert_rate(toy):
    labels, scores, _ = toy
    result = fixed_cutoff_policy(labels, scores, cutoff=0.9)
    assert result["alerts_per_100"] == pytest.approx(9.0)
    assert result["recall"] == pytest.approx(0.75)


def test_target_check_reports_the_gap():
    assert target_check(0.7, 0.8) == {
        "target": 0.8,
        "achieved_recall": 0.7,
        "gap": pytest.approx(0.1),
        "met": False,
    }
    assert target_check(0.85, 0.8)["met"] is True


def test_reliability_bins_compare_score_with_observed_rate():
    rng = np.random.default_rng(0)
    scores = rng.random(5000)
    labels = (rng.random(5000) < scores).astype(int)
    table = reliability_bins(labels, scores, bins=5)
    assert (table["observed_rate"] - table["mean_score"]).abs().max() < 0.06
