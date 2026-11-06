import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score  # noqa: E402

from src.config import (  # noqa: E402
    ALERT_BUDGET_SHARE,
    DATA_PATH,
    DEFAULT_SEED,
    MODEL_PATH,
    REPORTS_DIR,
    TARGET_RECALL,
    ZSCORE_CLIP_LIMIT,
)
from src.evaluation import (  # noqa: E402
    budget_needed_for_target,
    cluster_bootstrap_interval,
    fixed_cutoff_policy,
    flag_alerts,
    ranking_and_probability_metrics,
    recall_vs_alerts_curve,
    reliability_bins,
    target_check,
)
from src.features import DEVIATION_FEATURES, FEATURE_COLUMNS, build_features  # noqa: E402
from src.generate_mock_data import DIFFICULTIES, generate  # noqa: E402
from src.models import ModelBundle, new_aircraft_future_check  # noqa: E402
from src.schema import LABEL_COLUMN, SENSOR_NAMES, TIME_COLUMN, load_snapshots  # noqa: E402
from src.split import split_by_period  # noqa: E402
from src.train import train_bundle  # noqa: E402


def interval_dict(values: tuple[float, float, float]) -> dict[str, float]:
    estimate, low, high = values
    return {"estimate": estimate, "low": low, "high": high}


def evaluate_scores(
    part: pd.DataFrame, scores, budget_share: float = ALERT_BUDGET_SHARE, probabilities: bool = True
) -> dict:
    labels = part[LABEL_COLUMN].to_numpy()
    periods = part[TIME_COLUMN].to_numpy()
    alerts = flag_alerts(scores, periods, budget_share)
    bootstrap = cluster_bootstrap_interval(labels, alerts, part["component_id"].to_numpy())
    result = {
        "rows": len(part),
        "positives": int(labels.sum()),
        "alerts_per_100": float(100 * alerts.mean()),
    }
    for name in ("recall", "precision", "lift"):
        result[f"{name}_at_budget"] = interval_dict(bootstrap[name])
    metrics = (
        ranking_and_probability_metrics(labels, scores)
        if probabilities
        else {
            "pr_auc": float(average_precision_score(labels, scores)),
            "roc_auc": float(roc_auc_score(labels, scores)),
        }
    )
    result.update(metrics)
    return result


def alert_outcomes(
    snapshots: pd.DataFrame, part: pd.DataFrame, scores, budget_share: float
) -> dict:
    periods = part[TIME_COLUMN].to_numpy()
    flagged = flag_alerts(scores, periods, budget_share) == 1
    labels = part[LABEL_COLUMN].to_numpy() == 1
    ordered = snapshots.sort_values(["component_id", TIME_COLUMN], kind="mergesort")
    next_label = ordered.groupby("component_id")[LABEL_COLUMN].shift(-1).reindex(part.index)
    removal_next_window = (next_label == 1).to_numpy()
    # Only the final period has no known next window (a replaced component has no later row).
    checkable = periods < snapshots[TIME_COLUMN].max()
    false_alerts = flagged & ~labels
    checkable_alerts = flagged & checkable
    caught_within_60 = checkable_alerts & (labels | removal_next_window)
    by_period = {}
    for period in np.unique(periods):
        in_period = periods == period
        by_period[str(pd.Timestamp(period).date())] = {
            "removals": int(labels[in_period].sum()),
            "alerts": int(flagged[in_period].sum()),
            "removals_caught": int((labels & flagged)[in_period].sum()),
        }
    return {
        "true_positives": int((flagged & labels).sum()),
        "false_positives": int(false_alerts.sum()),
        "missed_removals": int((~flagged & labels).sum()),
        "true_negatives": int((~flagged & ~labels).sum()),
        "false_alerts_with_removal_in_cycles_31_to_60": int(
            (false_alerts & removal_next_window).sum()
        ),
        "false_alerts_without_removal_within_60_cycles": int(
            (false_alerts & checkable & ~removal_next_window).sum()
        ),
        "false_alerts_not_checkable_last_period": int((false_alerts & ~checkable).sum()),
        "precision_within_60_cycles": float(caught_within_60.sum() / checkable_alerts.sum()),
        "by_period": by_period,
    }


def reference_scores(
    part: pd.DataFrame, features: pd.DataFrame, seed: int
) -> dict[str, np.ndarray]:
    rows = features.loc[part.index]
    vibration = rows["vibration_mm_s_z_vs_type"]
    return {
        "random": np.random.default_rng(seed).random(len(part)),
        "cycles_since_new": rows["flight_cycles_since_new"].to_numpy(dtype=float),
        "latest_vibration_deviation": vibration.fillna(vibration.min()).to_numpy(dtype=float),
        # Fixed in advance, not tuned: how unstable all four sensors are versus similar parts.
        "sum_of_spread_deviations": rows[[f"{name}_spread_z_vs_type" for name in SENSOR_NAMES]]
        .fillna(0)
        .sum(axis=1)
        .to_numpy(dtype=float),
    }


def score_parts(
    bundle: ModelBundle, features: pd.DataFrame, parts: dict[str, pd.DataFrame]
) -> dict:
    scored = {}
    for part_name, part in parts.items():
        x = features.loc[part.index, FEATURE_COLUMNS]
        raw = {name: pipeline.predict_proba(x)[:, 1] for name, pipeline in bundle.pipelines.items()}
        scored[part_name] = {
            "raw": raw,
            "calibrated": bundle.calibrator.transform(raw[bundle.chosen_model_name]),
        }
    return scored


def data_preparation_summary(train_part: pd.DataFrame, features: pd.DataFrame) -> dict:
    sensor_columns = [f"{name}_mean" for name in SENSOR_NAMES]
    positives = int(train_part[LABEL_COLUMN].sum())
    rows = len(train_part)
    clipped = (
        (features.loc[train_part.index, DEVIATION_FEATURES].abs() > ZSCORE_CLIP_LIMIT).sum().sum()
    )
    return {
        "training_rows": rows,
        "training_positives": positives,
        "sensor_missing_share": train_part[sensor_columns].isna().mean().round(4).to_dict(),
        "deviation_values_clipped_for_logistic": int(clipped),
        "balanced_class_weight": {
            "positive": round(rows / (2 * positives), 2),
            "negative": round(rows / (2 * (rows - positives)), 2),
        },
        "category_levels": {
            column: int(train_part[column].nunique())
            for column in (
                "component_type",
                "part_number",
                "aircraft_type",
                "operator",
                "last_fault_code",
            )
        },
    }


def profile_summary(seed: int, aircraft_count: int = 80) -> dict:
    summary = {}
    for name in DIFFICULTIES:
        snapshots = generate(seed, name, aircraft_count)
        run = train_bundle(snapshots, seed=seed, tune=False)
        parts = {"test": run.split.test}
        scores = score_parts(run.bundle, run.features, parts)["test"]["calibrated"]
        result = evaluate_scores(run.split.test, scores)
        summary[name] = {
            "chosen_model": run.bundle.chosen_model_name,
            "recall_at_budget": result["recall_at_budget"],
            "alerts_per_100": result["alerts_per_100"],
            "test_positives": result["positives"],
            "target_check": target_check(result["recall_at_budget"]["estimate"]),
        }
    return summary


def build_report(
    snapshots: pd.DataFrame, bundle: ModelBundle, include_profiles: bool = True
) -> dict:
    split = split_by_period(snapshots)
    features = build_features(snapshots, bundle.baselines)
    parts = {"validation": split.validation, "test": split.test}
    scored = score_parts(bundle, features, parts)
    chosen = bundle.chosen_model_name

    models = {}
    for name in bundle.pipelines:
        models[name] = {}
        for part_name, part in parts.items():
            models[name][part_name] = evaluate_scores(part, scored[part_name]["raw"][name])
    for part_name, part in parts.items():
        calibrated = evaluate_scores(part, scored[part_name]["calibrated"])
        models[chosen][part_name]["brier_calibrated"] = calibrated["brier"]

    test = split.test
    test_labels = test[LABEL_COLUMN].to_numpy()
    test_periods = test[TIME_COLUMN].to_numpy()
    test_calibrated = scored["test"]["calibrated"]
    fixed = fixed_cutoff_policy(test_labels, test_calibrated, bundle.validation_cutoff_score)
    per_period_fixed = {
        str(pd.Timestamp(period).date()): float(
            100 * (test_calibrated[test_periods == period] >= bundle.validation_cutoff_score).mean()
        )
        for period in np.unique(test_periods)
    }
    top_five = models[chosen]["test"]

    curves, needed = {}, {}
    for part_name, part in parts.items():
        curves[part_name], needed[part_name] = {}, {}
        for name in bundle.pipelines:
            labels, periods = part[LABEL_COLUMN].to_numpy(), part[TIME_COLUMN].to_numpy()
            scores = scored[part_name]["raw"][name]
            curves[part_name][name] = recall_vs_alerts_curve(labels, scores, periods).to_dict(
                "records"
            )
            needed[part_name][name] = budget_needed_for_target(labels, scores, periods)

    references = {
        name: evaluate_scores(test, scores, probabilities=False)
        for name, scores in reference_scores(test, features, DEFAULT_SEED).items()
    }
    reliability = {
        "before_calibration": reliability_bins(test_labels, scored["test"]["raw"][chosen]).to_dict(
            "records"
        ),
        "after_calibration": reliability_bins(test_labels, test_calibrated).to_dict("records"),
    }
    achieved = top_five["recall_at_budget"]
    report = {
        "split": bundle.split_summary,
        "alert_budget_share": ALERT_BUDGET_SHARE,
        "chosen_model": chosen,
        "models": models,
        "policy_comparison": {
            "top_5_percent_per_period": {
                "recall": achieved["estimate"],
                "alerts_per_100": top_five["alerts_per_100"],
            },
            "fixed_validation_cutoff": {
                **fixed,
                "cutoff": bundle.validation_cutoff_score,
                "alerts_per_100_by_period": per_period_fixed,
            },
        },
        "alert_outcomes_top_5_percent": alert_outcomes(
            snapshots, test, scored["test"]["raw"][chosen], ALERT_BUDGET_SHARE
        ),
        "calibration": reliability,
        "recall_vs_alerts_curve": curves,
        "budget_needed_for_80_percent_recall": needed,
        "new_aircraft_future_check": new_aircraft_future_check(
            snapshots, bundle.parameters, bundle.seed
        ),
        "data_preparation": data_preparation_summary(split.train, features),
        "reference_baselines": references,
        "target_check": {
            **target_check(achieved["estimate"], TARGET_RECALL),
            "interval_low": achieved["low"],
            "interval_high": achieved["high"],
        },
    }
    if include_profiles:
        report["profiles"] = {
            "settings": "default settings, seed 42, 80 aircraft, test window recall at the budget",
            **profile_summary(DEFAULT_SEED),
        }
    return report


def clean_for_json(value):
    if isinstance(value, dict):
        return {key: clean_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_for_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def draw_charts(report: dict, snapshots: pd.DataFrame, bundle: ModelBundle) -> None:
    split = split_by_period(snapshots)
    features = build_features(snapshots, bundle.baselines)
    scored = score_parts(bundle, features, {"test": split.test})["test"]
    labels = split.test[LABEL_COLUMN].to_numpy()

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for axis, part_name in zip(axes, ("validation", "test"), strict=True):
        for name, rows in report["recall_vs_alerts_curve"][part_name].items():
            frame = pd.DataFrame(rows)
            axis.plot(frame.budget_per_100, frame.recall, marker="o", label=name)
        axis.axhline(TARGET_RECALL, color="red", linestyle="--", label="80% target")
        axis.axvline(5, color="grey", linestyle=":", label="5 alerts per 100")
        axis.set(
            title=f"Recall as the alert budget grows ({part_name})",
            xlabel="alerts per 100 components",
        )
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("recall")
    axes[1].legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "recall_vs_alerts.png", dpi=130)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.5, 4))
    for name, scores in scored["raw"].items():
        precision, recall, _ = precision_recall_curve(labels, scores)
        axis.plot(recall, precision, label=name)
    axis.axhline(labels.mean(), color="grey", linestyle=":", label="base rate")
    axis.set(title="Precision-recall on the test window", xlabel="recall", ylabel="precision")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "pr_curve.png", dpi=130)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.5, 4))
    for label, rows in report["calibration"].items():
        frame = pd.DataFrame(rows)
        axis.plot(frame.mean_score, frame.observed_rate, marker="o", label=label.replace("_", " "))
    top = max(axis.get_xlim()[1], axis.get_ylim()[1])
    axis.plot([0, top], [0, top], color="grey", linestyle=":", label="perfect")
    axis.set(
        title="Predicted risk against observed rate (test)",
        xlabel="mean score",
        ylabel="observed rate",
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "reliability.png", dpi=130)
    plt.close(figure)


def main() -> None:
    snapshots = load_snapshots(DATA_PATH)
    bundle = ModelBundle.load(MODEL_PATH)
    report = build_report(snapshots, bundle)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(clean_for_json(report), indent=2))
    draw_charts(report, snapshots, bundle)

    check = report["target_check"]
    chosen = report["chosen_model"]
    print(f"chosen model: {chosen}")
    for name, result in report["models"].items():
        test = result["test"]["recall_at_budget"]
        pr_auc = result["test"]["pr_auc"]
        print(
            f"  {name}: test recall at 5% alerts {test['estimate']:.2f} "
            f"(95% interval {test['low']:.2f} to {test['high']:.2f}), PR-AUC {pr_auc:.2f}"
        )
    verdict = "MET" if check["met"] else "NOT MET"
    print(f"80% target {verdict}: recall {check['achieved_recall']:.2f}, gap {check['gap']:.2f}")


if __name__ == "__main__":
    main()
