import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.config import BOOTSTRAP_RESAMPLES, DEFAULT_SEED, TARGET_RECALL


def alert_count(row_count: int, budget_share: float) -> int:
    # Rounding first keeps 0.05 x 400 from turning into 21 through float noise.
    return math.ceil(round(budget_share * row_count, 9))


def rank_within_period(scores, periods) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    periods = np.asarray(periods)
    ranks = np.empty(len(scores), dtype=int)
    for period in np.unique(periods):
        positions = np.flatnonzero(periods == period)
        order = np.argsort(-scores[positions], kind="stable")
        ranks[positions[order]] = np.arange(1, len(positions) + 1)
    return ranks


def flag_alerts(scores, periods, budget_share: float) -> np.ndarray:
    if not 0 <= budget_share <= 1:
        raise ValueError("The alert budget must be a share between 0 and 1")
    periods = np.asarray(periods)
    ranks = rank_within_period(scores, periods)
    alerts = np.zeros(len(ranks), dtype=int)
    for period in np.unique(periods):
        in_period = periods == period
        alerts[in_period] = ranks[in_period] <= alert_count(int(in_period.sum()), budget_share)
    return alerts


def recall_at_budget(labels, scores, periods, budget_share: float) -> float:
    labels = np.asarray(labels)
    alerts = flag_alerts(scores, periods, budget_share)
    positives = labels.sum()
    return float(labels[alerts == 1].sum() / positives) if positives else float("nan")


def precision_at_budget(labels, scores, periods, budget_share: float) -> float:
    labels = np.asarray(labels)
    alerts = flag_alerts(scores, periods, budget_share)
    return float(labels[alerts == 1].mean()) if alerts.sum() else float("nan")


def lift_at_budget(labels, scores, periods, budget_share: float) -> float:
    base_rate = float(np.asarray(labels).mean())
    return precision_at_budget(labels, scores, periods, budget_share) / base_rate


def cluster_bootstrap_interval(
    labels, alerts, component_ids, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = DEFAULT_SEED
) -> dict[str, tuple[float, float, float]]:
    """Recall, precision and lift with a 95% interval, resampling components (not rows)."""
    frame = pd.DataFrame(
        {
            "component": np.asarray(component_ids),
            "positive": np.asarray(labels, dtype=float),
            "alert": np.asarray(alerts, dtype=float),
        }
    )
    frame["caught"] = frame.positive * frame.alert
    frame["rows"] = 1.0
    per_component = frame.groupby("component")[["positive", "alert", "caught", "rows"]].sum()
    totals = per_component.to_numpy()
    draw = np.random.default_rng(seed).integers(0, len(totals), size=(resamples, len(totals)))
    resampled = totals[draw].sum(axis=1)
    positives, alert_total, caught, rows = (resampled[:, column] for column in range(4))

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = caught / positives
        precision = caught / alert_total
        lift = precision / (positives / rows)
    full = totals.sum(axis=0)
    estimates = {
        "recall": full[2] / full[0],
        "precision": full[2] / full[1],
        "lift": (full[2] / full[1]) / (full[0] / full[3]),
    }
    return {
        name: (
            float(estimates[name]),
            float(np.nanpercentile(values, 2.5)),
            float(np.nanpercentile(values, 97.5)),
        )
        for name, values in (("recall", recall), ("precision", precision), ("lift", lift))
    }


def recall_vs_alerts_curve(labels, scores, periods, budgets=range(1, 11)) -> pd.DataFrame:
    rows = []
    for budget_per_100 in budgets:
        share = budget_per_100 / 100
        alerts = flag_alerts(scores, periods, share)
        rows.append(
            {
                "budget_per_100": budget_per_100,
                "recall": recall_at_budget(labels, scores, periods, share),
                "precision": precision_at_budget(labels, scores, periods, share),
                "alerts_per_100": float(100 * alerts.mean()),
            }
        )
    return pd.DataFrame(rows)


def budget_needed_for_target(
    labels, scores, periods, target: float = TARGET_RECALL, max_share: float = 0.30
) -> float | None:
    for budget_per_100 in range(1, int(max_share * 100) + 1):
        share = budget_per_100 / 100
        if recall_at_budget(labels, scores, periods, share) >= target:
            return share
    return None


def fixed_cutoff_policy(labels, scores, cutoff: float) -> dict[str, float]:
    labels = np.asarray(labels)
    alerts = np.asarray(scores) >= cutoff
    positives = labels.sum()
    return {
        "recall": float(labels[alerts].sum() / positives) if positives else float("nan"),
        "alerts_per_100": float(100 * alerts.mean()),
        "precision": float(labels[alerts].mean()) if alerts.any() else float("nan"),
    }


def target_check(achieved_recall: float, target: float = TARGET_RECALL) -> dict:
    return {
        "target": target,
        "achieved_recall": achieved_recall,
        "gap": target - achieved_recall,
        "met": bool(achieved_recall >= target),
    }


def reliability_bins(labels, scores, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"label": np.asarray(labels), "score": np.asarray(scores)})
    frame["bin"] = pd.qcut(frame.score, bins, duplicates="drop")
    table = frame.groupby("bin", observed=True).agg(
        mean_score=("score", "mean"), observed_rate=("label", "mean"), rows=("label", "size")
    )
    return table.reset_index(drop=True)


def ranking_and_probability_metrics(labels, scores) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
    }
