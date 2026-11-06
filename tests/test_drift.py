import numpy as np
import pandas as pd

from src.drift import (
    drift_report,
    drift_report_by_group,
    ks_statistic,
    population_stability_index,
)


def samples(seed, shift=0.0, size=5000):
    return np.random.default_rng(seed).normal(shift, 1.0, size)


def test_psi_is_near_zero_for_the_same_distribution():
    assert population_stability_index(samples(1), samples(2)) < 0.02


def test_psi_is_large_for_a_mean_shift():
    assert population_stability_index(samples(1), samples(2, shift=1.5)) > 0.25


def test_ks_grows_with_the_shift():
    reference = samples(1)
    statistics = [ks_statistic(reference, samples(2, shift)) for shift in (0.0, 0.5, 1.0, 2.0)]
    assert statistics == sorted(statistics)
    assert statistics[-1] > statistics[0] + 0.5


def test_nan_values_are_ignored():
    reference, current = samples(1), samples(2, shift=1.0)
    with_nan = np.concatenate([current, [np.nan] * 100])
    assert population_stability_index(reference, with_nan) == population_stability_index(
        reference, current
    )
    assert ks_statistic(pd.Series(reference), pd.Series(with_nan)) == ks_statistic(
        reference, current
    )


def test_empty_input_gives_nan():
    values = samples(1)
    assert np.isnan(population_stability_index(values, np.array([])))
    assert np.isnan(population_stability_index(np.array([np.nan]), values))
    assert np.isnan(ks_statistic(values, np.array([])))
    assert np.isnan(ks_statistic(np.array([np.nan]), values))


def test_tied_quantiles_do_not_break_psi():
    reference = np.array([0.0] * 90 + [1.0] * 10)
    assert population_stability_index(reference, reference) == 0.0


def make_frames(seed_offset=0):
    rng = np.random.default_rng(10 + seed_offset)
    reference = pd.DataFrame({"a": rng.normal(size=3000), "b": rng.normal(size=3000)})
    new = pd.DataFrame({"a": rng.normal(size=3000), "b": rng.normal(2.0, 1.0, size=3000)})
    return reference, new


def test_drift_report_puts_the_shifted_column_first():
    reference, new = make_frames()
    report = drift_report(reference, new)
    assert list(report.columns) == ["feature", "psi", "ks"]
    assert report.loc[0, "feature"] == "b"
    assert report["psi"].is_monotonic_decreasing
    assert len(drift_report(reference, new, top_n=1)) == 1


def test_drift_report_by_group_shows_drift_only_in_the_shifted_group():
    rng = np.random.default_rng(5)
    reference = pd.DataFrame({"group": ["x", "y"] * 2000, "value": rng.normal(size=4000)})
    new_values = rng.normal(size=4000)
    new = pd.DataFrame({"group": ["x", "y"] * 2000, "value": new_values})
    new.loc[new["group"] == "y", "value"] += 2.0
    report = drift_report_by_group(reference, new, "group")
    psi_by_group = report.set_index("group")["psi"]
    assert psi_by_group["y"] > 0.25
    assert psi_by_group["x"] < 0.05


def test_groups_missing_from_either_frame_are_skipped():
    reference, new = make_frames()
    reference["group"] = ["x", "y"] * 1500
    new["group"] = ["x", "z"] * 1500
    report = drift_report_by_group(reference, new, "group")
    assert set(report["group"]) == {"x"}
