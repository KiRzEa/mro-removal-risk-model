import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import ALERT_BUDGET_SHARE, MODEL_PATH, TARGET_RECALL
from src.explain import explain_rows
from src.models import ModelBundle, score_snapshots
from src.schema import LABEL_COLUMN, TIME_COLUMN, load_snapshots

OUTPUT_COLUMNS = [
    "component_id",
    TIME_COLUMN,
    "risk_score",
    "rank_in_period",
    "alert",
    "confidence",
    "reason_1",
    "reason_2",
    "reason_3",
]


def score_file(input_path: Path, model_path: Path, explain_all: bool = False) -> pd.DataFrame:
    snapshots = load_snapshots(input_path)
    if not model_path.exists():
        raise ValueError(f"Model file not found: {model_path}")
    bundle = ModelBundle.load(model_path)

    scored = score_snapshots(bundle, snapshots)
    reasons = pd.DataFrame("", index=scored.index, columns=["reason_1", "reason_2", "reason_3"])
    explained_rows = list(scored.index if explain_all else scored.index[scored.alert == 1])
    if explained_rows:
        for index, row_reasons in explain_rows(bundle, snapshots, explained_rows).items():
            reasons.loc[index] = row_reasons
    return pd.concat([scored, reasons], axis=1)[OUTPUT_COLUMNS]


def grade_scores(labels: pd.Series, scored: pd.DataFrame) -> dict:
    removals = int(labels.sum())
    flagged = scored.alert.to_numpy() == 1
    caught = int((flagged & (labels.to_numpy() == 1)).sum())
    alerts_per_100 = float(100 * flagged.mean())
    recall = caught / removals if removals else float("nan")
    return {
        "rows": len(scored),
        "removals": removals,
        "alerts_per_100": alerts_per_100,
        "removals_caught": caught,
        "recall_at_alert_budget": recall,
        "precision_at_alert_budget": caught / int(flagged.sum()) if flagged.any() else float("nan"),
        "pr_auc": float(average_precision_score(labels, scored.risk_score)) if removals else None,
        "meets_target": bool(
            recall >= TARGET_RECALL and alerts_per_100 <= 100 * ALERT_BUDGET_SHARE + 1e-9
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a snapshot file with the saved model")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=Path("scores.csv"))
    parser.add_argument("--explain-all", action="store_true", help="give reasons for every row")
    parser.add_argument(
        "--metrics-output", type=Path, help="write the grading summary here as JSON (needs labels)"
    )
    args = parser.parse_args(argv)

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = score_file(args.input, args.model, args.explain_all)
        for warning in caught:
            if "extra columns" in str(warning.message):
                print(f"warning: {warning.message}", file=sys.stderr)
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    result.to_csv(args.output, index=False)
    flagged = int(result.alert.sum())
    low = int((result.confidence == "low").sum())
    print(
        f"scored {len(result)} rows, {flagged} alerts, {low} low-confidence rows -> {args.output}"
    )

    labels = load_snapshots(args.input).get(LABEL_COLUMN)
    if labels is not None and labels.notna().all():
        grade = grade_scores(labels.astype(int), result)
        print(
            f"grading on {grade['removals']} removals: "
            f"recall {grade['recall_at_alert_budget']:.2f} "
            f"at {grade['alerts_per_100']:.1f} alerts per 100, "
            f"precision {grade['precision_at_alert_budget']:.2f}, PR-AUC {grade['pr_auc']:.2f}; "
            f"meets the 80% / 5-per-100 target: {'yes' if grade['meets_target'] else 'no'}"
        )
        if args.metrics_output:
            args.metrics_output.write_text(json.dumps(grade, indent=2))
    elif args.metrics_output:
        print("warning: no label column, so there is nothing to grade", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
