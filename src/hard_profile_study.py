import dataclasses
import json
import os

import numpy as np

from src import generate_mock_data as generator
from src.config import REPORTS_DIR
from src.evaluate import clean_for_json
from src.evaluation import recall_at_budget
from src.models import score_snapshots
from src.schema import LABEL_COLUMN, TIME_COLUMN
from src.stability import run_pipeline
from src.train import train_bundle

SEEDS = range(1, 9)
BUDGETS_PER_100 = (5, 8, 10, 15, 20)


def variants() -> dict:
    hard = generator.DIFFICULTIES["hard"]
    return {
        "hard as defined": hard,
        "no sudden failures": dataclasses.replace(hard, sudden_failure_share=0.0),
        "sudden failures at the realistic level (15%)": dataclasses.replace(
            hard, sudden_failure_share=0.15
        ),
        "missing values at the realistic level (6%)": dataclasses.replace(hard, missing_rate=0.06),
        "no label noise": dataclasses.replace(hard, label_drop_rate=0.0),
        "degradation signal at the realistic level (3.0)": dataclasses.replace(
            hard, precursor_strength=3.0
        ),
    }


def recall_by_budget(seed: int) -> dict[int, float]:
    snapshots = generator.generate(seed, "hard", 80)
    run = train_bundle(snapshots, seed=seed, tune=False)
    test = run.split.test
    scores = score_snapshots(run.bundle, snapshots).loc[test.index, "risk_score"].to_numpy()
    labels, periods = test[LABEL_COLUMN].to_numpy(), test[TIME_COLUMN].to_numpy()
    return {
        budget: recall_at_budget(labels, scores, periods, budget / 100)
        for budget in BUDGETS_PER_100
    }


def main() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    original = generator.DIFFICULTIES["hard"]
    report = {
        "seeds": list(SEEDS),
        "settings": "untuned, hard profile, test window",
        "ablations": {},
    }
    try:
        for name, difficulty in variants().items():
            generator.DIFFICULTIES["hard"] = difficulty
            recalls = [
                run_pipeline(generator.generate(seed, "hard", 80), seed, tune=False)[
                    "chosen_recall"
                ]
                for seed in SEEDS
            ]
            report["ablations"][name] = {
                "mean_recall": float(np.mean(recalls)),
                "min": float(min(recalls)),
                "max": float(max(recalls)),
            }
            print(f"{name:50s} mean recall {np.mean(recalls):.2f}")

        generator.DIFFICULTIES["hard"] = original
        per_seed = [recall_by_budget(seed) for seed in SEEDS]
        report["recall_by_alerts_per_100"] = {
            str(budget): float(np.mean([row[budget] for row in per_seed]))
            for budget in BUDGETS_PER_100
        }
        for budget, value in report["recall_by_alerts_per_100"].items():
            print(f"hard, {budget} alerts per 100: mean recall {value:.2f}")
    finally:
        generator.DIFFICULTIES["hard"] = original

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "hard_profile_study.json").write_text(
        json.dumps(clean_for_json(report), indent=2)
    )


if __name__ == "__main__":
    main()
