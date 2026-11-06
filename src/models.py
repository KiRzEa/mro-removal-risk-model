from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import (
    ALERT_BUDGET_SHARE,
    DEFAULT_SEED,
    MAX_MISSING_SENSOR_SHARE,
    MIN_HISTORY_FOR_NORMAL_CONFIDENCE,
    ZSCORE_CLIP_LIMIT,
)
from src.evaluation import (
    cluster_bootstrap_interval,
    flag_alerts,
    rank_within_period,
)
from src.features import (
    CATEGORICAL_FEATURES,
    DEVIATION_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    FleetBaselines,
    build_features,
    fit_fleet_baselines,
)
from src.schema import LABEL_COLUMN, SENSOR_NAMES, TIME_COLUMN
from src.split import aircraft_folds, split_by_period

LOGISTIC = "logistic_regression"
BOOSTING = "gradient_boosting"

PLAIN_NUMERIC_FEATURES = [name for name in NUMERIC_FEATURES if name not in DEVIATION_FEATURES]


def build_logistic_pipeline(regularization: float = 1.0, seed: int = DEFAULT_SEED) -> Pipeline:
    clip = FunctionTransformer(
        np.clip, kw_args={"a_min": -ZSCORE_CLIP_LIMIT, "a_max": ZSCORE_CLIP_LIMIT}
    )
    deviations = make_pipeline(
        clip, SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
    )
    plain = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
    )
    prepare = ColumnTransformer(
        [
            ("deviations", deviations, DEVIATION_FEATURES),
            ("plain", plain, PLAIN_NUMERIC_FEATURES),
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = LogisticRegression(
        C=regularization, class_weight="balanced", max_iter=2000, random_state=seed
    )
    return Pipeline([("prepare", prepare), ("model", model)])


def build_boosting_pipeline(
    learning_rate: float = 0.05, max_depth: int = 4, max_iter: int = 200, seed: int = DEFAULT_SEED
) -> Pipeline:
    categories = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan
    )
    prepare = ColumnTransformer(
        [
            ("numbers", "passthrough", NUMERIC_FEATURES),
            ("categories", categories, CATEGORICAL_FEATURES),
        ]
    )
    first_category = len(NUMERIC_FEATURES)
    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_iter=max_iter,
        class_weight="balanced",
        categorical_features=list(
            range(first_category, first_category + len(CATEGORICAL_FEATURES))
        ),
        random_state=seed,
    )
    return Pipeline([("prepare", prepare), ("model", model)])


@dataclass
class ModelBundle:
    baselines: FleetBaselines
    pipelines: dict[str, Pipeline]
    chosen_model_name: str
    calibrator: object | None = None
    validation_cutoff_score: float | None = None
    alert_budget_share: float = ALERT_BUDGET_SHARE
    parameters: dict = field(default_factory=dict)
    split_summary: dict = field(default_factory=dict)
    library_versions: dict = field(default_factory=dict)
    seed: int = DEFAULT_SEED

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "ModelBundle":
        return joblib.load(path)


def predict_risk(bundle: ModelBundle, features: pd.DataFrame, model_name: str | None = None):
    name = model_name or bundle.chosen_model_name
    raw_scores = bundle.pipelines[name].predict_proba(features[FEATURE_COLUMNS])[:, 1]
    if bundle.calibrator is not None and name == bundle.chosen_model_name:
        return bundle.calibrator.transform(raw_scores)
    return raw_scores


def score_snapshots(bundle: ModelBundle, snapshots: pd.DataFrame) -> pd.DataFrame:
    features = build_features(snapshots, bundle.baselines)
    scores = predict_risk(bundle, features)
    periods = snapshots[TIME_COLUMN].to_numpy()
    return pd.DataFrame(
        {
            "component_id": snapshots["component_id"],
            TIME_COLUMN: snapshots[TIME_COLUMN],
            "risk_score": scores,
            "rank_in_period": rank_within_period(scores, periods),
            "alert": flag_alerts(scores, periods, bundle.alert_budget_share),
            "confidence": confidence_flags(bundle, snapshots, features),
        },
        index=snapshots.index,
    )


def confidence_flags(bundle: ModelBundle, snapshots: pd.DataFrame, features: pd.DataFrame):
    """Low when the row has little history, mostly missing sensors, or a type never seen."""
    missing_share = features[[f"{name}_is_missing" for name in SENSOR_NAMES]].mean(axis=1)
    known_type = snapshots["component_type"].isin(bundle.baselines.table.index)
    is_low = (
        (features["history_length"] < MIN_HISTORY_FOR_NORMAL_CONFIDENCE)
        | (missing_share > MAX_MISSING_SENSOR_SHARE)
        | ~known_type
    )
    return np.where(is_low, "low", "normal")


BUILDERS = {LOGISTIC: build_logistic_pipeline, BOOSTING: build_boosting_pipeline}
DEFAULT_PARAMETERS = {
    LOGISTIC: {"regularization": 1.0},
    BOOSTING: {"learning_rate": 0.05, "max_depth": 4, "max_iter": 200},
}
PARAMETER_GRIDS = {
    LOGISTIC: [{"regularization": value} for value in (0.1, 1.0, 10.0)],
    BOOSTING: [
        {"learning_rate": rate, "max_depth": depth, "max_iter": iterations}
        for rate, depth, iterations in (
            (0.05, 3, 150),
            (0.05, 4, 200),
            (0.1, 3, 100),
            (0.1, 4, 150),
        )
    ],
}
FOLD_STARTS = (9, 11, 13)
FOLD_VALIDATION_PERIODS = 2


def build_pipeline(name: str, parameters: dict, seed: int) -> Pipeline:
    return BUILDERS[name](**parameters, seed=seed)


def forward_chaining_folds(train_snapshots: pd.DataFrame):
    dates = np.sort(train_snapshots[TIME_COLUMN].unique())
    for start in FOLD_STARTS:
        when = train_snapshots[TIME_COLUMN]
        fold_train = train_snapshots[when < dates[start]]
        end = start + FOLD_VALIDATION_PERIODS
        in_window = when >= dates[start]
        if end < len(dates):
            in_window &= when < dates[end]
        fold_validation = train_snapshots[in_window]
        yield fold_train, fold_validation


def tune_models(train_snapshots: pd.DataFrame, seed: int = DEFAULT_SEED) -> dict[str, dict]:
    """Pick settings by average precision on folds that validate on later periods."""
    prepared = []
    for fold_train, fold_validation in forward_chaining_folds(train_snapshots):
        baselines = fit_fleet_baselines(fold_train)
        # History for validation rows comes from earlier snapshots, never later ones.
        rows = pd.concat([fold_train, fold_validation])
        features = build_features(rows, baselines)
        prepared.append(
            (
                features.loc[fold_train.index, FEATURE_COLUMNS],
                fold_train[LABEL_COLUMN],
                features.loc[fold_validation.index, FEATURE_COLUMNS],
                fold_validation[LABEL_COLUMN],
            )
        )

    best = {}
    for name, grid in PARAMETER_GRIDS.items():
        results = []
        for parameters in grid:
            fold_scores = []
            for x_train, y_train, x_validation, y_validation in prepared:
                pipeline = build_pipeline(name, parameters, seed).fit(x_train, y_train)
                scores = pipeline.predict_proba(x_validation)[:, 1]
                fold_scores.append(average_precision_score(y_validation, scores))
            results.append((float(np.mean(fold_scores)), parameters))
        best[name] = max(results, key=lambda item: item[0])[1]
    return best


class SigmoidCalibrator:
    def fit(self, raw_scores, labels) -> "SigmoidCalibrator":
        self.model = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(raw_scores), labels)
        return self

    def transform(self, raw_scores) -> np.ndarray:
        return self.model.predict_proba(self._logit(raw_scores))[:, 1]

    @staticmethod
    def _logit(scores) -> np.ndarray:
        clipped = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def new_aircraft_future_check(
    snapshots: pd.DataFrame,
    parameters: dict[str, dict],
    seed: int = DEFAULT_SEED,
    fold_count: int = 5,
    partition_seeds: int = 5,
    budget_share: float = ALERT_BUDGET_SHARE,
) -> dict[str, dict]:
    """Train on earlier periods of some aircraft, score later periods of aircraft never trained on.

    Settings were tuned beforehand on all aircraft: this checks the fitted models, not the tuning.
    Scores from all folds are pooled and ranked fleet-wide in each period, like the main evaluation.
    """
    split = split_by_period(snapshots)
    results = {name: {"pooled_by_partition": []} for name in BUILDERS}
    for partition_seed in range(seed, seed + partition_seeds):
        pooled = {name: {"scores": [], "rows": []} for name in BUILDERS}
        for held_out in aircraft_folds(snapshots, fold_count, partition_seed):
            train_rows = split.train[~split.train["aircraft_id"].isin(held_out)]
            test_rows = split.test[split.test["aircraft_id"].isin(held_out)]
            features = build_features(snapshots, fit_fleet_baselines(train_rows))
            for name in BUILDERS:
                pipeline = build_pipeline(name, parameters[name], seed)
                pipeline.fit(
                    features.loc[train_rows.index, FEATURE_COLUMNS], train_rows[LABEL_COLUMN]
                )
                pooled[name]["scores"].append(
                    pipeline.predict_proba(features.loc[test_rows.index, FEATURE_COLUMNS])[:, 1]
                )
                pooled[name]["rows"].append(test_rows)
        for name, parts in pooled.items():
            rows = pd.concat(parts["rows"])
            scores = np.concatenate(parts["scores"])
            labels = rows[LABEL_COLUMN].to_numpy()
            alerts = flag_alerts(scores, rows[TIME_COLUMN].to_numpy(), budget_share)
            recall, low, high = cluster_bootstrap_interval(
                labels, alerts, rows["component_id"].to_numpy()
            )["recall"]
            results[name]["pooled_by_partition"].append(
                {"recall": recall, "low": low, "high": high, "positives": int(labels.sum())}
            )
    for result in results.values():
        recalls = [entry["recall"] for entry in result["pooled_by_partition"]]
        result["main_partition"] = result["pooled_by_partition"][0]
        result["recall_range_over_partitions"] = [min(recalls), max(recalls)]
    return results
