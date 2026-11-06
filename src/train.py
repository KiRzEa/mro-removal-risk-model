import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score

from src.config import ALERT_BUDGET_SHARE, DATA_PATH, DEFAULT_SEED, MODEL_PATH, REPORTS_DIR
from src.evaluation import recall_at_budget
from src.features import FEATURE_COLUMNS, build_features, fit_fleet_baselines
from src.models import (
    BUILDERS,
    DEFAULT_PARAMETERS,
    ModelBundle,
    SigmoidCalibrator,
    build_pipeline,
    score_snapshots,
    tune_models,
)
from src.schema import LABEL_COLUMN, TIME_COLUMN, load_snapshots
from src.split import Split, split_by_period


@dataclass
class TrainedRun:
    bundle: ModelBundle
    split: Split
    features: pd.DataFrame
    validation_results: dict[str, dict[str, float]]
    best_parameters: dict[str, dict]


def choose_model(validation_results: dict[str, dict[str, float]]) -> str:
    # Recall at the alert budget decides; average precision breaks ties.
    return max(
        validation_results,
        key=lambda name: (
            validation_results[name]["recall_at_budget"],
            validation_results[name]["average_precision"],
        ),
    )


def summarize_split(split: Split) -> dict:
    return {
        name: {
            "rows": len(part),
            "positives": int(part[LABEL_COLUMN].sum()),
            "first_period": str(part[TIME_COLUMN].min().date()),
            "last_period": str(part[TIME_COLUMN].max().date()),
        }
        for name, part in zip(("train", "validation", "test"), split, strict=True)
    }


def train_bundle(
    snapshots: pd.DataFrame, seed: int = DEFAULT_SEED, tune: bool = True, split: Split | None = None
) -> TrainedRun:
    split = split or split_by_period(snapshots)
    best_parameters = tune_models(split.train, seed) if tune else DEFAULT_PARAMETERS

    baselines = fit_fleet_baselines(split.train)
    # Features come from the full history; each row only reads earlier rows.
    features = build_features(snapshots, baselines)
    train_x = features.loc[split.train.index, FEATURE_COLUMNS]
    pipelines = {
        name: build_pipeline(name, best_parameters[name], seed).fit(
            train_x, split.train[LABEL_COLUMN]
        )
        for name in BUILDERS
    }

    validation_x = features.loc[split.validation.index, FEATURE_COLUMNS]
    validation_labels = split.validation[LABEL_COLUMN].to_numpy()
    validation_periods = split.validation[TIME_COLUMN].to_numpy()
    raw_validation_scores = {}
    validation_results = {}
    for name, pipeline in pipelines.items():
        scores = pipeline.predict_proba(validation_x)[:, 1]
        raw_validation_scores[name] = scores
        validation_results[name] = {
            "recall_at_budget": recall_at_budget(
                validation_labels, scores, validation_periods, ALERT_BUDGET_SHARE
            ),
            "average_precision": float(average_precision_score(validation_labels, scores)),
        }
    chosen = choose_model(validation_results)

    calibrator = SigmoidCalibrator().fit(raw_validation_scores[chosen], validation_labels)
    calibrated = calibrator.transform(raw_validation_scores[chosen])
    bundle = ModelBundle(
        baselines=baselines,
        pipelines=pipelines,
        chosen_model_name=chosen,
        calibrator=calibrator,
        validation_cutoff_score=float(np.quantile(calibrated, 1 - ALERT_BUDGET_SHARE)),
        parameters=best_parameters,
        split_summary=summarize_split(split),
        library_versions={
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        seed=seed,
    )
    return TrainedRun(bundle, split, features, validation_results, best_parameters)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the removal risk models")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-tuning", action="store_true", help="use default settings")
    args = parser.parse_args()

    snapshots = load_snapshots(DATA_PATH)
    run = train_bundle(snapshots, seed=args.seed, tune=not args.no_tuning)
    run.bundle.save(MODEL_PATH)

    REPORTS_DIR.mkdir(exist_ok=True)
    scored = score_snapshots(run.bundle, snapshots)
    test_scores = scored.loc[run.split.test.index].copy()
    test_scores[LABEL_COLUMN] = run.split.test[LABEL_COLUMN]
    test_scores.to_csv(REPORTS_DIR / "test_scores.csv", index=False)

    for name, result in run.validation_results.items():
        print(
            f"{name}: validation recall at {ALERT_BUDGET_SHARE:.0%} alerts "
            f"{result['recall_at_budget']:.2f}, "
            f"average precision {result['average_precision']:.2f}, "
            f"settings {run.best_parameters[name]}"
        )
    alerts = test_scores.groupby(TIME_COLUMN).alert.agg(["sum", "size"])
    print(f"chosen model: {run.bundle.chosen_model_name}")
    per_100 = (100 * alerts["sum"] / alerts["size"]).tolist()
    print(f"test alerts per 100 components in each period: {per_100}")
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
