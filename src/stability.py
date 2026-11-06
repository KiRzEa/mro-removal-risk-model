import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from src.config import ALERT_BUDGET_SHARE, REPORTS_DIR, TARGET_RECALL  # noqa: E402
from src.evaluate import clean_for_json, reference_scores, score_parts  # noqa: E402
from src.evaluation import recall_at_budget  # noqa: E402
from src.generate_mock_data import generate  # noqa: E402
from src.schema import LABEL_COLUMN, TIME_COLUMN  # noqa: E402
from src.train import train_bundle  # noqa: E402

REQUIRED_RUNS = 20


def summarize_runs(values, target: float = TARGET_RECALL, minimum_runs: int = 1) -> dict:
    requested = len(values)
    if requested < minimum_runs:
        raise ValueError(f"The study needs at least {minimum_runs} runs, got {requested}")
    valid = np.array([value for value in values if not np.isnan(value)], dtype=float)
    reaching = int((valid >= target).sum())
    share_low, share_high = wilson_interval(reaching, len(valid))
    mean_error = (
        stats.t.ppf(0.975, len(valid) - 1) * valid.std(ddof=1) / np.sqrt(len(valid))
        if len(valid) > 1
        else float("nan")
    )
    return {
        "runs_requested": requested,
        "runs_used": int(len(valid)),
        "mean": float(valid.mean()),
        "mean_interval_95": [float(valid.mean() - mean_error), float(valid.mean() + mean_error)],
        "std": float(valid.std(ddof=1)) if len(valid) > 1 else float("nan"),
        "median": float(np.median(valid)),
        "p2_5": float(np.percentile(valid, 2.5)),
        "p97_5": float(np.percentile(valid, 97.5)),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "share_at_or_above_target": float((valid >= target).mean()),
        "runs_at_or_above_target": reaching,
        "share_interval_95": [share_low, share_high],
    }


def summarize_values(values) -> dict:
    valid = np.array(values, dtype=float)
    error = stats.t.ppf(0.975, len(valid) - 1) * valid.std(ddof=1) / np.sqrt(len(valid))
    return {
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=1)),
        "mean_interval_95": [float(valid.mean() - error), float(valid.mean() + error)],
        "min": float(valid.min()),
        "max": float(valid.max()),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    share = successes / total
    centre = (share + z**2 / (2 * total)) / (1 + z**2 / total)
    half_width = (
        z * np.sqrt(share * (1 - share) / total + z**2 / (4 * total**2)) / (1 + z**2 / total)
    )
    return float(centre - half_width), float(centre + half_width)


def paired_p_value(better, worse) -> float:
    # One-sided Wilcoxon signed-rank test on the per-seed differences.
    differences = np.asarray(better, dtype=float) - np.asarray(worse, dtype=float)
    if np.all(differences == 0):
        return 1.0
    return float(stats.wilcoxon(differences, alternative="greater").pvalue)


def run_pipeline(snapshots: pd.DataFrame, seed: int, tune: bool) -> dict:
    run = train_bundle(snapshots, seed=seed, tune=tune)
    test = run.split.test
    labels = test[LABEL_COLUMN].to_numpy()
    periods = test[TIME_COLUMN].to_numpy()
    scored = score_parts(run.bundle, run.features, {"test": test})["test"]
    recalls = {
        name: recall_at_budget(labels, scores, periods, ALERT_BUDGET_SHARE)
        for name, scores in scored["raw"].items()
    }
    pr_aucs = {
        name: float(average_precision_score(labels, scores))
        for name, scores in scored["raw"].items()
    }
    references = {
        name: recall_at_budget(labels, scores, periods, ALERT_BUDGET_SHARE)
        for name, scores in reference_scores(test, run.features, seed).items()
    }
    chosen = run.bundle.chosen_model_name
    return {
        "test_positives": int(labels.sum()),
        "chosen_model": chosen,
        "recall_by_model": recalls,
        "pr_auc_by_model": pr_aucs,
        "chosen_recall": recalls[chosen],
        "reference_recall": references,
        "best_reference_recall": max(references.values()),
    }


def run_pipeline_for_seed(
    seed: int, difficulty: str, tune: bool = True, aircraft: int = 80
) -> dict:
    snapshots = generate(seed, difficulty, aircraft)
    return {"seed": seed, **run_pipeline(snapshots, seed, tune)}


def summarize_study(per_seed: list[dict], minimum_runs: int) -> dict:
    model_names = list(per_seed[0]["recall_by_model"])
    summary = {
        name: summarize_runs(
            [run["recall_by_model"][name] for run in per_seed], TARGET_RECALL, minimum_runs
        )
        for name in model_names
    }
    summary["pr_auc"] = {
        name: summarize_values([run["pr_auc_by_model"][name] for run in per_seed])
        for name in model_names
    }
    summary["chosen_model"] = summarize_runs(
        [run["chosen_recall"] for run in per_seed], TARGET_RECALL, minimum_runs
    )
    summary["best_reference_ranking"] = summarize_runs(
        [run["best_reference_recall"] for run in per_seed], TARGET_RECALL, minimum_runs
    )
    summary["reference_rankings"] = {
        name: summarize_runs(
            [run["reference_recall"][name] for run in per_seed], TARGET_RECALL, minimum_runs
        )
        for name in per_seed[0]["reference_recall"]
    }
    gains = [run["chosen_recall"] - run["best_reference_recall"] for run in per_seed]
    summary["mean_gain_over_best_reference"] = float(np.nanmean(gains))
    chosen_recalls = [run["chosen_recall"] for run in per_seed]
    summary["p_values"] = {
        "chosen_beats_best_simple_rule_wilcoxon_one_sided": paired_p_value(
            chosen_recalls, [run["best_reference_recall"] for run in per_seed]
        ),
    }
    if {"gradient_boosting", "logistic_regression"} <= set(model_names):
        summary["p_values"]["boosting_beats_logistic_wilcoxon_one_sided"] = paired_p_value(
            [run["recall_by_model"]["gradient_boosting"] for run in per_seed],
            [run["recall_by_model"]["logistic_regression"] for run in per_seed],
        )
        summary["p_values"]["boosting_beats_logistic_pr_auc_wilcoxon_one_sided"] = paired_p_value(
            [run["pr_auc_by_model"]["gradient_boosting"] for run in per_seed],
            [run["pr_auc_by_model"]["logistic_regression"] for run in per_seed],
        )
    return summary


def draw_chart(per_seed: list[dict], summary: dict, difficulty: str, suffix: str = "") -> None:
    names = [name for name in summary if name in per_seed[0]["recall_by_model"]]
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for position, name in enumerate(names):
        values = [run["recall_by_model"][name] for run in per_seed]
        axes[0].scatter(
            np.full(len(values), position) + np.linspace(-0.15, 0.15, len(values)),
            values,
            alpha=0.8,
        )
    references = [run["best_reference_recall"] for run in per_seed]
    axes[0].scatter(
        np.full(len(references), len(names)) + np.linspace(-0.15, 0.15, len(references)),
        references,
        alpha=0.8,
    )
    axes[0].axhline(TARGET_RECALL, color="red", linestyle="--")
    axes[0].set_xticks(range(len(names) + 1))
    axes[0].set_xticklabels([*names, "best simple rule"], fontsize=8)
    axes[0].set(ylabel="recall at 5% alert budget", title=f"Test recall per seed ({difficulty})")
    axes[0].grid(alpha=0.3)

    chosen = [run["chosen_recall"] for run in per_seed]
    axes[1].hist(chosen, bins=10, color="grey")
    axes[1].axvline(TARGET_RECALL, color="red", linestyle="--", label="80% target")
    share = summary["chosen_model"]["share_at_or_above_target"]
    axes[1].set(
        title=f"Chosen model: {share:.0%} of seeds reach 80%", xlabel="recall at 5% alert budget"
    )
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "stability.png", dpi=130)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat the pipeline over many generated datasets")
    parser.add_argument("--seeds", type=int, default=REQUIRED_RUNS)
    parser.add_argument("--difficulty", default="realistic", choices=["easy", "realistic", "hard"])
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    # One thread per worker: the pool already uses every core, and extra threads only contend.
    os.environ["OMP_NUM_THREADS"] = "1"
    seeds = list(range(1, args.seeds + 1))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        per_seed = list(pool.map(run_pipeline_for_seed, seeds, [args.difficulty] * len(seeds)))
    quick = args.seeds < REQUIRED_RUNS
    summary = summarize_study(per_seed, minimum_runs=1 if quick else REQUIRED_RUNS)
    report = {
        "settings": {
            "seeds": seeds,
            "difficulty": args.difficulty,
            "tuning": True,
            "quick_mode_fewer_than_required_seeds": quick,
            "uncertainty_note": "Two sources: sampling noise inside one dataset (the component "
            "bootstrap in metrics.json) and randomness of the simulation across seeds (this file).",
        },
        "per_seed": per_seed,
        "summary": summary,
    }
    suffix = "" if args.difficulty == "realistic" else f"_{args.difficulty}"
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / f"stability{suffix}.json").write_text(
        json.dumps(clean_for_json(report), indent=2)
    )
    draw_chart(per_seed, summary, args.difficulty, suffix)

    for name, values in summary.items():
        if isinstance(values, dict) and "mean" in values:
            share = values["share_at_or_above_target"]
            print(
                f"{name}: mean {values['mean']:.2f}, 2.5-97.5 percentile {values['p2_5']:.2f} "
                f"to {values['p97_5']:.2f}, share of seeds at or above 0.80: {share:.0%}"
            )
    gain = summary["mean_gain_over_best_reference"]
    print(f"mean gain of the chosen model over the best simple rule: {gain:.2f}")


if __name__ == "__main__":
    main()
