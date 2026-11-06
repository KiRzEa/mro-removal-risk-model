import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from src.config import DATA_PATH, MODEL_PATH, REPORTS_DIR  # noqa: E402
from src.features import FEATURE_COLUMNS, build_features  # noqa: E402
from src.models import BOOSTING, ModelBundle, score_snapshots  # noqa: E402
from src.schema import LABEL_COLUMN, SENSOR_NAMES, TIME_COLUMN, load_snapshots  # noqa: E402
from src.split import split_by_period  # noqa: E402

SENSOR_LABELS = {
    "vibration_mm_s": "Vibration",
    "oil_temp_c": "Oil temperature",
    "pressure_kpa": "Pressure",
    "current_a": "Current draw",
}
FALLBACK_REASON = "No further factor stands out"


def feature_label(name: str) -> str:
    for sensor, label in SENSOR_LABELS.items():
        if name.startswith(sensor):
            suffix = name[len(sensor) + 1 :]
            return f"{label} ({suffix.replace('_', ' ')})"
    return name.replace("_", " ")


def above_or_below(value: float) -> str:
    return "above" if value >= 0 else "below"


def describe_feature(name: str, value) -> str:
    sensor = next((key for key in SENSOR_NAMES if name.startswith(key)), None)
    label = SENSOR_LABELS.get(sensor, "")
    if pd.isna(value):
        return f"{feature_label(name).capitalize()} is not reported"

    if sensor:
        suffix = name[len(sensor) + 1 :]
        if suffix == "z_vs_type":
            return f"{label} is well {above_or_below(value)} the level of similar components"
        if suffix == "spread_z_vs_type":
            return f"{label} readings are {'more' if value >= 0 else 'less'} unstable than usual"
        if suffix == "z_change_vs_previous":
            return f"{label} {'rose' if value >= 0 else 'fell'} since the previous snapshot"
        if suffix == "z_slope_last_3":
            direction = "up" if value >= 0 else "down"
            return f"{label} has been trending {direction} over the last 3 snapshots"
        if suffix == "z_vs_own_prior_mean":
            return f"{label} is {above_or_below(value)} this component's own recent level"
        if suffix == "is_missing":
            return f"The {label.lower()} sensor stopped reporting"
        if suffix in ("mean", "std"):
            return f"{label} {'level' if suffix == 'mean' else 'variation'} is unusual"

    count = int(value) if isinstance(value, (int, float, np.integer, np.floating)) else value
    templates = {
        "fault_count_last_30_cycles": f"{count} fault messages in the last 30 cycles",
        "distinct_fault_codes_last_30_cycles": f"{count} different fault codes recently",
        "fault_count_prior_3": f"{count} fault messages in the 3 snapshots before",
        "high_risk_fault_recent": "High-risk fault codes (F415 or F520) were logged recently",
        "last_fault_code": f"The latest fault code is {value}",
        "cycles_since_last_fault": f"The last fault message came {count} cycles ago",
        "flight_cycles_since_new": f"The component has flown {count:,} cycles since new"
        if isinstance(count, int)
        else "The component is old",
        "flight_hours_since_new": "The component has many flight hours since new",
        "cycles_since_overhaul": f"{count} cycles since the last overhaul"
        if isinstance(count, int)
        else "Long since the last overhaul",
        "maintenance_actions_last_90_days": f"{count} maintenance actions in the last 90 days",
        "days_since_last_maintenance": f"{count} days since the last maintenance action",
        "prior_unscheduled_removals_in_slot": f"{count} earlier removals in this position",
        "history_length": f"Only {count} earlier snapshots are available for this component",
        "flight_hours_last_30_cycles": "Flight hours in the last 30 cycles are unusual",
        "component_type": f"Component type {value} is removed more often",
        "part_number": f"Part number {value} is removed more often",
        "aircraft_type": f"Aircraft type {value} shows more removals",
        "operator": f"Operator {value} shows more removals",
    }
    return templates.get(name, f"{feature_label(name).capitalize()} is unusual")


SENSOR_SIGNAL_KINDS = {
    "z_vs_type": "level",
    "mean": "level",
    "spread_z_vs_type": "spread",
    "std": "spread",
    "z_change_vs_previous": "trend",
    "z_slope_last_3": "trend",
    "z_vs_own_prior_mean": "trend",
    "is_missing": "missing",
}


def signal_kind(name: str) -> str:
    for sensor in SENSOR_NAMES:
        if name.startswith(sensor):
            return SENSOR_SIGNAL_KINDS.get(name[len(sensor) + 1 :], name)
    return name


def top_reasons(feature_row: pd.Series, contribution_row: pd.Series, count: int = 3) -> list[str]:
    # One reason per kind of signal, so three sensors drifting together read as one story.
    chosen, kinds_seen = [], set()
    for name in contribution_row.sort_values(ascending=False).index:
        if signal_kind(name) not in kinds_seen:
            kinds_seen.add(signal_kind(name))
            chosen.append(name)
        if len(chosen) == count:
            break
    reasons = [describe_feature(name, feature_row[name]) for name in chosen]
    reasons += [FALLBACK_REASON] * (count - len(reasons))
    return reasons


def contributions(bundle: ModelBundle, features: pd.DataFrame) -> pd.DataFrame:
    """Per-feature push on the risk, in log-odds, from the gradient boosting model."""
    pipeline = bundle.pipelines[BOOSTING]
    transformed = pipeline.named_steps["prepare"].transform(features[FEATURE_COLUMNS])
    values = shap.TreeExplainer(pipeline.named_steps["model"]).shap_values(transformed)
    return pd.DataFrame(np.asarray(values), index=features.index, columns=FEATURE_COLUMNS)


def explain_rows(
    bundle: ModelBundle, snapshots: pd.DataFrame, row_indexes: list, count: int = 3
) -> dict:
    features = build_features(snapshots, bundle.baselines)
    selected = features.loc[row_indexes]
    values = contributions(bundle, selected)
    return {
        index: top_reasons(selected.loc[index], values.loc[index], count) for index in row_indexes
    }


def global_importance(
    bundle: ModelBundle, features: pd.DataFrame, sample_size: int = 500
) -> pd.Series:
    sample = features.sample(min(sample_size, len(features)), random_state=0)
    return contributions(bundle, sample).abs().mean().sort_values(ascending=False)


CHART_SUFFIX_LABELS = {
    "z_vs_type": "vs similar components",
    "spread_z_vs_type": "variability vs similar components",
    "z_change_vs_previous": "change since last month",
    "z_slope_last_3": "trend over 3 months",
    "z_vs_own_prior_mean": "vs its own recent level",
    "is_missing": "missing",
    "mean": "average",
    "std": "variability",
}


def chart_label(name: str) -> str:
    for sensor, label in SENSOR_LABELS.items():
        if name.startswith(sensor):
            suffix = name[len(sensor) + 1 :]
            return f"{label}: {CHART_SUFFIX_LABELS.get(suffix, suffix.replace('_', ' '))}"
    return name.replace("_", " ")


def draw_global_importance(importance: pd.Series) -> None:
    top = importance.head(15).iloc[::-1]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh([chart_label(name) for name in top.index], top.values, color="#333333")
    axis.set(
        title="Main drivers of the risk score",
        xlabel="Influence on the risk score (higher = more risk)",
    )
    axis.grid(alpha=0.3, axis="x")
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "global_importance.png", dpi=130)
    plt.close(figure)


def pick_worked_examples(scored: pd.DataFrame) -> dict[str, int]:
    alerted = scored[scored.alert == 1]
    positives = scored[scored[LABEL_COLUMN] == 1]
    true_positive = int(alerted[alerted[LABEL_COLUMN] == 1].risk_score.idxmax())
    false_alerts = alerted[
        (alerted[LABEL_COLUMN] == 0)
        & (alerted.component_id != scored.at[true_positive, "component_id"])
    ]
    return {
        "true_positive": true_positive,
        "false_positive": int(false_alerts.risk_score.idxmax()),
        "missed_positive": int(positives[positives.alert == 0].risk_score.idxmin()),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.to_numpy()]
    return "\n".join([header, divider, *rows])


def recent_history(snapshots: pd.DataFrame, features: pd.DataFrame, index: int) -> pd.DataFrame:
    component = snapshots.at[index, "component_id"]
    rows = snapshots[snapshots.component_id == component].sort_values(TIME_COLUMN)
    rows = rows[rows.index <= index].tail(4)
    table = pd.DataFrame(
        {
            "period": rows[TIME_COLUMN].dt.strftime("%Y-%m-%d"),
            **{
                f"{SENSOR_LABELS[name].split()[0].lower()} z": (
                    features.loc[rows.index, f"{name}_z_vs_type"].round(1) + 0.0
                )
                for name in SENSOR_NAMES
            },
            "faults": rows["fault_count_last_30_cycles"],
            "last code": rows["last_fault_code"],
        }
    )
    return table.astype(object).where(table.notna(), "missing")


def reading_for(kind: str, snapshots: pd.DataFrame, features: pd.DataFrame, index: int) -> str:
    component = snapshots.at[index, "component_id"]
    later_removal = snapshots[
        (snapshots.component_id == component)
        & (snapshots[TIME_COLUMN] > snapshots.at[index, TIME_COLUMN])
        & (snapshots[LABEL_COLUMN] == 1)
    ]
    sensor_columns = [f"{name}_z_vs_type" for name in SENSOR_NAMES]
    largest = features.loc[index, sensor_columns].abs().max()
    if kind == "true_positive":
        return (
            "The model ranked this component near the top of its period and it was removed within "
            "the next 30 cycles, as predicted. The reasons match the history below."
        )
    if kind == "false_positive":
        if len(later_removal):
            when = (
                later_removal[TIME_COLUMN].iloc[0] - snapshots.at[index, TIME_COLUMN]
            ).days // 30
            unit = "period" if when == 1 else "periods"
            return (
                "This alert was not followed by a removal in the next 30 cycles, but the part was "
                f"removed about {when} {unit} later. By the strict 30-cycle label this is a false "
                "alert, but the warning came before the removal. Such alerts are the cost of the "
                "5% budget."
            )
        return (
            "This alert was not followed by a removal in the next 30 cycles and no later one is "
            "in the data. The history looked degraded, the kind of false alert the budget accepts."
        )
    return (
        "The model did not flag this removal. Its largest sensor deviation was "
        f"{largest:.1f} standard deviations from similar components, so it looked close to normal; "
        "removals without a visible warning cap how much any model can catch."
    )


def write_examples(snapshots, bundle, scored, examples, features, contribution_values) -> None:
    titles = {
        "true_positive": "True positive: flagged and removed",
        "false_positive": "False positive: flagged but not removed in time",
        "missed_positive": "Missed positive: removed but not flagged",
    }
    lines = [
        "# Example predictions and explanations",
        "",
        "Scores are calibrated risk estimates; reasons come from the gradient boosting model. "
        "All data is synthetic.",
        "",
        "Reasons are signals the model relied on, not confirmed causes: recent maintenance, for "
        "example, may follow a problem instead of causing it. In the history tables a value of +2 "
        "means two standard deviations above the average for the same component type, and a "
        "negative value means below it.",
        "",
    ]
    for kind, index in examples.items():
        row = scored.loc[index]
        reasons = top_reasons(features.loc[index], contribution_values.loc[index])
        lines += [
            f"## {titles[kind]}",
            "",
            f"Component {row.component_id}, period {row[TIME_COLUMN].date()}: risk score "
            f"{row.risk_score:.3f}, rank {int(row.rank_in_period)} in its period, "
            f"{'flagged' if row.alert else 'not flagged'}.",
            "",
            "Strongest risk signals (none strong enough to flag):"
            if kind == "missed_positive"
            else "Top reasons:",
            "",
            *[f"{number}. {reason}" for number, reason in enumerate(reasons, start=1)],
            "",
            "Last four snapshots (sensor values as standard deviations from similar components):",
            "",
            markdown_table(recent_history(snapshots, features, index)),
            "",
            reading_for(kind, snapshots, features, index),
            "",
        ]
    (REPORTS_DIR / "example_predictions.md").write_text("\n".join(lines))


def main() -> None:
    snapshots = load_snapshots(DATA_PATH)
    bundle = ModelBundle.load(MODEL_PATH)
    split = split_by_period(snapshots)
    features = build_features(snapshots, bundle.baselines)
    scored = score_snapshots(bundle, snapshots)
    scored[LABEL_COLUMN] = snapshots[LABEL_COLUMN]
    test_scored = scored.loc[split.test.index]

    REPORTS_DIR.mkdir(exist_ok=True)
    importance = global_importance(bundle, features.loc[split.validation.index])
    draw_global_importance(importance)
    examples = pick_worked_examples(test_scored)
    values = contributions(bundle, features.loc[list(examples.values())])
    write_examples(snapshots, bundle, test_scored, examples, features, values)

    print("top drivers:", ", ".join(feature_label(name) for name in importance.head(6).index))
    print("worked examples:", json.dumps({kind: int(index) for kind, index in examples.items()}))


if __name__ == "__main__":
    main()
