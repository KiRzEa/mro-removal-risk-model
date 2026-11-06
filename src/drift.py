import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

MIN_PROPORTION = 1e-4


def _clean(values):
    array = np.asarray(values, dtype=float)
    return array[~np.isnan(array)]


# Rule of thumb for PSI: below 0.1 is small, 0.1 to 0.25 moderate, above 0.25 large.
def population_stability_index(reference, current, bins=10):
    reference, current = _clean(reference), _clean(current)
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    inner_edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    edges = np.concatenate([[-np.inf], inner_edges, [np.inf]])
    expected = np.histogram(reference, edges)[0] / len(reference)
    actual = np.histogram(current, edges)[0] / len(current)
    expected = np.clip(expected, MIN_PROPORTION, None)
    actual = np.clip(actual, MIN_PROPORTION, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def ks_statistic(reference, current):
    reference, current = _clean(reference), _clean(current)
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    return float(ks_2samp(reference, current).statistic)


def drift_report(reference_features, new_features, top_n=10):
    numeric_columns = reference_features.select_dtypes("number").columns
    shared_columns = [column for column in numeric_columns if column in new_features.columns]
    rows = [
        {
            "feature": column,
            "psi": population_stability_index(reference_features[column], new_features[column]),
            "ks": ks_statistic(reference_features[column], new_features[column]),
        }
        for column in shared_columns
    ]
    report = pd.DataFrame(rows, columns=["feature", "psi", "ks"])
    return report.sort_values("psi", ascending=False).head(top_n).reset_index(drop=True)


def drift_report_by_group(reference_features, new_features, group_column, top_n=10):
    reference_groups = dict(tuple(reference_features.groupby(group_column)))
    new_groups = dict(tuple(new_features.groupby(group_column)))
    reports = []
    for group, reference_rows in reference_groups.items():
        if group not in new_groups:
            continue
        report = drift_report(
            reference_rows.drop(columns=group_column),
            new_groups[group].drop(columns=group_column),
            top_n,
        )
        reports.append(report.assign(group=group))
    if not reports:
        return pd.DataFrame(columns=["group", "feature", "psi", "ks"])
    return pd.concat(reports, ignore_index=True)[["group", "feature", "psi", "ks"]]
