import json
from dataclasses import asdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import DATA_PATH, DEFAULT_SEED, MODEL_PATH, REPORTS_DIR, TARGET_RECALL  # noqa: E402
from src.drift import drift_report, drift_report_by_group  # noqa: E402
from src.evaluate import clean_for_json, evaluate_scores  # noqa: E402
from src.evaluation import fixed_cutoff_policy  # noqa: E402
from src.features import build_features  # noqa: E402
from src.generate_mock_data import Shift, generate  # noqa: E402
from src.models import ModelBundle, score_snapshots  # noqa: E402
from src.schema import LABEL_COLUMN, load_snapshots  # noqa: E402
from src.split import split_by_period  # noqa: E402

FRESH_SEEDS = (101, 102, 103, 104, 105)
UNSEEN_TYPE = "brake_assembly"
# These grow with the calendar by construction, so their PSI always looks like drift.
TIME_TRENDING_FEATURES = [
    "history_length",
    "flight_cycles_since_new",
    "flight_hours_since_new",
    "cycles_since_overhaul",
]
SHIFT_SCENARIOS = {
    "sensor_offset_one_operator": Shift(sensor_offset_sd=1.5),
    "higher_usage": Shift(utilization_factor=1.3),
    "more_missing_data": Shift(extra_missing_rate=0.2),
    "low_base_rate": Shift(life_scale_factor=2.0),
    "high_base_rate": Shift(life_scale_factor=0.5),
    "faster_degradation": Shift(precursor_factor=0.5),
    "early_failures_of_new_components": Shift(early_failure_share=0.3),
}


def evaluate_scenario(
    name: str,
    snapshots: pd.DataFrame,
    bundle: ModelBundle,
    reference_features: pd.DataFrame,
    settings,
) -> dict:
    split = split_by_period(snapshots)
    test = split.test
    features = build_features(snapshots, bundle.baselines)
    # Scored on the full frame so each row keeps the history it would have in production.
    scores = score_snapshots(bundle, snapshots).loc[test.index, "risk_score"].to_numpy()
    result = evaluate_scores(test, scores)
    fixed = fixed_cutoff_policy(
        test[LABEL_COLUMN].to_numpy(), scores, bundle.validation_cutoff_score
    )
    test_features = features.loc[test.index].drop(columns=TIME_TRENDING_FEATURES)
    reference_features = reference_features.drop(columns=TIME_TRENDING_FEATURES)
    return {
        "name": name,
        "settings": settings,
        "test_rows": result["rows"],
        "test_positives": result["positives"],
        "test_positive_rate": result["positives"] / result["rows"],
        "recall_at_budget": result["recall_at_budget"],
        "alerts_per_100_fixed_cutoff": fixed["alerts_per_100"],
        "recall_fixed_cutoff": fixed["recall"],
        "psi_pooled_top": drift_report(reference_features, test_features, top_n=5).to_dict(
            "records"
        ),
        "psi_by_operator_top": drift_report_by_group(
            reference_features, test_features, "operator", top_n=3
        ).to_dict("records"),
    }


def draw_summary(scenarios: list[dict]) -> None:
    names = [scenario["name"].replace("_", " ") for scenario in scenarios]
    recalls = [scenario["recall_at_budget"] for scenario in scenarios]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    estimates = [entry["estimate"] for entry in recalls]
    errors = [
        [e["estimate"] - e["low"] for e in recalls],
        [e["high"] - e["estimate"] for e in recalls],
    ]
    axes[0].barh(names, estimates, xerr=errors, color="grey", capsize=3)
    axes[0].axvline(TARGET_RECALL, color="red", linestyle="--", label="80% target")
    axes[0].set(title="Recall at the 5% alert budget (95% interval)", xlim=(0, 1.05))
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].barh(names, [s["alerts_per_100_fixed_cutoff"] for s in scenarios], color="grey")
    axes[1].axvline(5, color="red", linestyle="--", label="5 per 100 budget")
    axes[1].set(title="Alerts per 100 with the fixed validation cutoff")
    axes[1].set_yticklabels([])
    axes[1].legend(loc="lower right", fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.3, axis="x")
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "stress_summary.png", dpi=130)
    plt.close(figure)


def main() -> None:
    original = load_snapshots(DATA_PATH)
    bundle = ModelBundle.load(MODEL_PATH)
    train_rows = split_by_period(original).train
    reference_features = build_features(original, bundle.baselines).loc[train_rows.index]

    scenarios = [
        evaluate_scenario("no_shift_committed_dataset", original, bundle, reference_features, {})
    ]
    for name, shift in SHIFT_SCENARIOS.items():
        shifted = generate(DEFAULT_SEED, "realistic", 80, shift=shift)
        scenarios.append(
            evaluate_scenario(name, shifted, bundle, reference_features, asdict(shift))
        )

    for seed in FRESH_SEEDS:
        fresh = generate(seed, "realistic", 80)
        result = evaluate_scenario(
            f"fresh_seed_{seed}", fresh, bundle, reference_features, {"seed": seed}
        )
        result["note"] = "Same settings, new random seed: a generalization check, not drift."
        scenarios.append(result)

    one_unseen = original.assign(
        component_type=original.component_type.replace(UNSEEN_TYPE, "brand_new_type")
    )
    result = evaluate_scenario(
        "one_unseen_component_type",
        one_unseen,
        bundle,
        reference_features,
        {"renamed": UNSEEN_TYPE},
    )
    result["note"] = f"{UNSEEN_TYPE} is unknown to the model, the other types are not."
    scenarios.append(result)

    all_unseen = original.assign(component_type="brand_new_type")
    result = evaluate_scenario(
        "all_component_types_unseen", all_unseen, bundle, reference_features, {"renamed": "all"}
    )
    result["note"] = "No type baselines exist at all: the harshest cold start."
    scenarios.append(result)

    report = {
        "caveat": "The model is trained on the committed dataset. Every scenario uses the same "
        "generator, so robustness is shown only for the shifts that were modelled.",
        "scenarios": scenarios,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "stress_test.json").write_text(json.dumps(clean_for_json(report), indent=2))
    draw_summary(scenarios)

    for scenario in scenarios:
        recall = scenario["recall_at_budget"]
        print(
            f"{scenario['name']:36s} recall {recall['estimate']:.2f} "
            f"({recall['low']:.2f}-{recall['high']:.2f}), positives {scenario['test_positives']}, "
            f"alerts/100 with fixed cutoff {scenario['alerts_per_100_fixed_cutoff']:.1f}"
        )


if __name__ == "__main__":
    main()
