import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

from src.config import MODEL_PATH
from src.explain import explain_rows
from src.models import ModelBundle, score_snapshots
from src.schema import TIME_COLUMN, load_snapshots

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a snapshot file with the saved model")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=Path("scores.csv"))
    parser.add_argument("--explain-all", action="store_true", help="give reasons for every row")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
