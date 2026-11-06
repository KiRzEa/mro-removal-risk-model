# Component removal risk model

A proof of concept that predicts whether an installed aircraft component needs an **unscheduled removal within
the next 30 flight cycles**, with a mock dataset, a time-split evaluation, explanations, drift stress tests and a
command to score your own file. All data is **synthetic**, produced by `src/generate_mock_data.py`, so results show
how the method behaves on modelled data, not how it would behave on a real fleet.

## Results at a glance

Alert rule: flag the top 5% of active components by score in each monthly period (5 alerts per 100 components).

| Evidence | Recall at the 5% budget | Detail |
|---|---|---|
| Committed dataset, final test window (34 removals) | **0.88** (95% interval 0.76 to 0.97) | above the 0.80 target on this draw, interval reaches below it |
| 20 regenerated datasets, whole pipeline repeated | mean **0.84** (range 0.61 to 0.94) | reaches 0.80 on **16 of 20** datasets |
| Best simple ranking rule (same seeds) | mean 0.74 | the model adds about 0.11 |

The committed dataset is one draw from a random generator with only 34 test removals, so the
seed study is the headline claim and the single result is shown with its interval. The data is synthetic and easy by
construction: a no-training rule that sums the four sensor-spread deviations reaches 0.74 on average. See `docs/REPORT.md` for the
full write-up and `reports/` for the raw numbers and charts.

## Run it

```bash
uv sync                                    # install the locked dependencies
uv run python -m src.generate_mock_data    # data/mock_components.csv and data/SCHEMA.md (already committed)
uv run python -m src.train                 # tune, fit both models, calibrate -> models/model.joblib
uv run python -m src.evaluate              # reports/metrics.json and charts
uv run python -m src.stress_test           # reports/stress_test.json
uv run python -m src.stability             # reports/stability.json and .png (20 seeds, realistic profile; per-seed recall and PR-AUC)
uv run python -m src.stability --difficulty easy   # reports/stability_easy.json; also --difficulty hard
uv run python -m src.explain               # reports/example_predictions.md
uv run pytest                              # 81 offline tests
```

`--seeds 5` is a quick check but overwrites `reports/stability.json`. Training takes about 20 seconds, evaluation about 10 seconds, the 20-seed study about 2 minutes on a laptop.

## Score your own file

Build a CSV that follows `data/SCHEMA.md` (the generator with another seed works) and run:

```bash
uv run python -m src.score --input your_file.csv --output scores.csv
```

The label column is optional. When it is present the command also grades the file and prints recall at the alert
budget, precision, alerts per 100, PR-AUC and whether the 80% recall / 5 alerts per 100 target is met; add
`--metrics-output grade.json` to save that summary. Every row of the file is graded, so use a file the model has not
seen (for example `generate_mock_data` with another seed). The output has one row per input row, in the input order:
`component_id, snapshot_date, risk_score, rank_in_period, alert, confidence, reason_1, reason_2, reason_3`.

- `alert` is 1 for the top 5% of rows in each distinct `snapshot_date`, rounded up, so a date with fewer than 20 rows still gets one alert. Score the whole fleet and its full history together.
- `confidence` is `low` when the component has fewer than 2 earlier snapshots, more than half of its sensors are
  missing, or its component type was not seen in training.
- Reasons are filled for alerted rows (every row with `--explain-all`).
- Unseen components, unseen category values and missing values are scored. Missing columns, unreadable dates,
  text in numeric columns, an empty file or duplicate (`component_id`, `snapshot_date`) rows stop the command with a
  message that names the problem.

## Layout

```
src/        schema, generator, features, split, models, evaluation, evaluate, train, stability,
            stress_test, explain, drift, score
tests/      generator, features (no look-ahead), split, evaluation, stability, explain, drift, scoring contract
data/       mock_components.csv and SCHEMA.md
models/     model.joblib (committed so a file can be scored without training)
reports/    metrics.json, stability*.json, stress_test.json, charts,
            example_predictions.md, test_scores.csv
docs/       REPORT.md (write-up) and Problem5-Slides.pptx / .pdf (the slide deck)
```

## Honest limits

- Synthetic data from our own generator: robustness is shown only for the shifts the generator models.
- The test window has about 34 removals, so any single recall figure carries a wide interval.
- A component type the model has never seen is scored poorly (recall 0.65 when one of five types is unknown, 0.18 when all are): it gets a low-confidence flag and needs baselines before it can be trusted.
