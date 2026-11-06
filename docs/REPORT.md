# Problem 5 - Machine Learning Solution: report

Written to match the slide deck `docs/Problem5-Slides.pptx`: same order, same terms, same numbers. All data is synthetic,
produced by `src/generate_mock_data.py`. Numbers come from `reports/*.json`; commands to regenerate them are in `README.md`.

## 1. Problem and target

Warn maintenance engineers which components are likely to be pulled off an aircraft soon.

- A **component** is an installed aircraft part, such as a hydraulic pump. An **unscheduled removal** is a component taken
  off without a plan; we predict it for the next **30 flight cycles** (one cycle = one takeoff and landing; about a month of
  flying in this simulation).
- An **alert** is a component flagged for inspection. Each month every component gets a risk score and the top 5% are flagged.
- Target from the brief: about 2% of records are removals; at most **5 alerts per 100 active components** each month; at least
  **80% of actual removals detected**. Catching 80% of removals with only 5% flagged means about 1 alert in 3 is a real
  removal, so many false alarms are expected.

## 2. Mock dataset (synthetic)

No real data was shared, so we built a simulator.

- 10,000 records: one per component every 30 flight cycles (shown as months): 80 aircraft x 5 components x 25 months; 2% are removals.
- Each record: usage, 4 sensor readings, fault messages, maintenance history. A hidden wear-out process makes sensors drift
  before a removal. Label: 1 if the component is removed unplanned within 30 flight cycles.

| What we vary | Easy | Realistic | Hard |
|---|---|---|---|
| Removals with no warning signs | 5% | 15% | 35% |
| Strength of warning signs | strong | medium | weak |
| Missing sensor readings | 2% | 6% | 15% |
| Mislabelled removals | 0% | 1.5% | 4% |
| Fleet changes in later months | none | mild | strong |

Three difficulty levels, all with about 2% removals. **Realistic is the headline**, fixed before any model was run. The dataset
shipped in `data/mock_components.csv` is one fixed draw of it (seed 42).

## 3. Dataset schema

One row = one component at one monthly snapshot; features look back, the label looks 30 cycles ahead. 29 columns (28 inputs and
the label); the full list with units is `data/SCHEMA.md`.

| Group | Key columns | Values |
|---|---|---|
| Identity and time | component_id, aircraft_id, snapshot_date | Serial number; 80 aircraft; 25 monthly snapshots |
| Component and fleet | component_type, aircraft_type, operator, part_number | 5 component types (hydraulic pump, fuel pump, starter-generator, bleed-air valve, brake assembly); A320, B737, ATR72; airlines OP_A, OP_B, OP_C |
| Usage | flight_cycles_since_new, flight_hours_since_new, cycles_since_overhaul | How much the component has flown |
| Sensors (can be missing) | vibration, oil temperature, pressure, current: mean and std over the last 30 cycles | 8 columns |
| Faults and maintenance | fault_count, last_fault_code, cycles_since_last_fault, days_since_last_maintenance | Fault messages and maintenance history |
| Label | removal_within_30_cycles | 1 = removed without a plan within the next 30 cycles; about 2% are 1 |

## 4. Data preparation and splitting

**Split by time: learn from the past, test on the future.**

| Set | Share of records | Months | Removals | Role |
|---|---|---|---|---|
| Train | 60% | 1-15 (Jan 2022 - Feb 2023) | 124 | teaches the model |
| Validation | 20% | 16-20 (Mar - Jul 2023) | 45 | picks the model |
| Test | 20% | 21-25 (Aug - Dec 2023) | 34 | final exam, used once |

- **Missing readings**: marked as missing (it is itself a warning sign), with an is-missing flag per sensor; gaps are filled with a
  typical value where needed (median from training for the logistic model; boosting reads missing values natively).
- **Extreme values**: kept (they are often real failures); limited to a ceiling for the simple (logistic) model only.
- **Categories**: component type, airline and similar labels are turned into numbers the models can read (one-hot for logistic,
  native categorical for boosting; unknown values are ignored or treated as missing). Identifiers and dates are never inputs.
- **Few removals (class imbalance)**: only 2% are removals, so removals get extra weight in training (balanced class weights) and we
  judge by removals caught, not accuracy. Oversampling (SMOTE) was not used: it would invent trajectories that do not exist.
- **No leakage**: about 200 removals in total. Each record uses only earlier months of the same component; baselines, imputers,
  scalers and encoders are fitted on training rows only; the test months are never used to make choices. A test changes every
  later row and checks that earlier features do not move. A separate check on unseen aircraft (train on some aircraft, test on
  others, five groupings pooled fleet-wide) gives recall 0.91 to 0.97 for boosting; the groupings reuse the same 34 test removals,
  so it shows recall does not fall without an aircraft, not that the model works on new fleets.

## 5. How the model decides

50 features per component per month, all computed from the current and earlier months of the same component.

| Engineered feature (x4 sensors) | Count | How it is computed |
|---|---|---|
| oil_temp_c_z_vs_type | 4 | (this month's reading - average of the same component type) / spread of that type |
| oil_temp_c_spread_z_vs_type | 4 | Same, for the within-month spread (std) of the reading |
| oil_temp_c_z_change_vs_previous | 4 | Deviation now minus deviation last month |
| oil_temp_c_z_slope_last_3 | 4 | (Deviation now - deviation 2 months ago) / 2 |
| oil_temp_c_z_vs_own_prior_mean | 4 | Deviation now minus this component's own average of its last 3 months |
| fault_count_prior_3, high_risk_fault_recent, history_length, oil_temp_c_is_missing | 7 | Faults in the 3 months before, serious code (F415/F520) in last 3 months, months of history, sensor missing flag |
| Copied as they are | 23 | Age, cycles, hours, faults, maintenance, raw sensor mean/std (18), plus 5 labels (component type, part number, aircraft type, airline, last fault code) |

The strongest are the deviation-from-type features; change since last month turned out weak. A new component with no history
gets missing history features and a low-confidence flag.

## 6. The two models compared

| | Logistic regression | Gradient boosting (chosen) |
|---|---|---|
| How it works | Adds up weighted clues into one score | Many small decision trees, each fixing the last one's mistakes |
| Strength | Simple, stable, easy to explain | Finds complex patterns and combinations of clues |
| Weakness | Misses complex patterns | Harder to explain, so we add plain-language reasons for each alert |
| Missing and extreme values | Filled with a typical value; extremes capped | Handled directly; extremes kept as they are |

Both models get the same clues and the same split, and each has a small tuning grid (3 settings for logistic regression, 4 for
boosting) tuned on forward-chaining folds inside the training months. They are compared over 20 seeds in section 8, and the
chosen model is reported in section 9. Selection uses validation recall at the 5% budget, with average precision as tie-breaker;
boosting was chosen on every one of the 20 seeds.

## 7. Evaluation metrics and alert threshold

| Metric | What it measures | How we use it |
|---|---|---|
| Recall@5% alerts (primary) | Share of true removals that fall in the top 5% highest-risk components of their month | Target >= 0.80, from the brief |
| Precision@5% alerts | Share of alerts that were real removals | Cost of false alarms for inspectors |
| PR-AUC | Area under the precision-recall curve: ranking quality over all cut-offs (random = base rate, about 0.02) | Compares models; suited to rare positives |
| 95% CI and 20 seeds | Bootstrap over components (1,000 resamples); repeat on 20 independent seeds | Only 34 test removals, so show uncertainty |

- Accuracy is not used: always predicting "no removal" is 98% accurate and catches nothing.
- **Alert rule (all results)**: top 5% by risk score each month = 20 of 400 components; the model is chosen on validation months
  and the test months are used once. The capacity limit holds by construction.
- **Fixed cutoff (drift signal)**: score 0.035, the top-5% level on the validation months, held fixed. On the test months it gives
  3.95 alerts per 100 (5.25, 3.75, 5.0, 2.25 and 3.5 by month) and the same 0.88 recall. It is not the operating rule, but its alert
  volume moves when the fleet changes, so a shift in alerts per 100 signals drift; a capped rule always shows 5 and hides it.
- Scores are calibrated probabilities (sigmoid fit on the validation months); reliability bins before and after are in
  `reports/metrics.json` under `calibration`. Calibration does not change the ranking, so the alert list is unchanged.

## 8. Which model is better

Compared only over **20 independent seeds** (each regenerates the dataset and repeats tuning, training and test scoring), because
one dataset with 34 test removals is too noisy to separate the models. Both are evaluated identically: same time split, features,
tuning method, metrics and alert rule.

| 20 seeds | Logistic regression | Gradient boosting (chosen) | Reading |
|---|---|---|---|
| Recall@5%, mean +- std | 0.74 +- 0.09 | 0.84 +- 0.08 | Boosting about +0.10 |
| 95% CI of the mean recall | 0.70-0.78 | 0.81-0.88 | Intervals do not overlap |
| Seeds with recall >= 0.80 | 5/20 | 16/20 | Target met far more often |
| Recall: higher model per seed (paired) | 0 seeds | 18 seeds (2 tied) | Wilcoxon p < 0.001 |
| PR-AUC, mean +- std | 0.43 +- 0.08 | 0.70 +- 0.07 | Boosting ranks removals better |
| PR-AUC: higher model per seed (paired) | 0 seeds | 20 seeds (0 tied) | Wilcoxon p < 0.001 |

- Gradient boosting is better on both recall@5% (about +0.10) and ranking quality, and wins in most seeds, so it is the chosen model.
- Logistic regression usually misses the 0.80 target (5 of 20 seeds); boosting meets it in 16 of 20, but its worst seed is 0.61:
  usually met, not guaranteed.
- A simple untrained rule (sum of the four sensor-spread deviations, fixed in advance) also scores about 0.74, so linear clues
  plateau there and the trees add about 0.10 by combining clues. This is our reading of the numbers; it was not tested separately.
- The standard deviation of about 0.08 per seed is why a single split such as seed 42 cannot rank the models (there logistic
  regression caught 31 of 34 removals and boosting 30, a tie within noise).
- The p-values say the gap is not luck across simulated datasets, nothing about real fleets.

## 9. Results of the chosen model (20 seeds)

Test months 21-25 of each seed: 400 components x 5 months = 2,000 component-months, with a fixed alert budget of 20 per month =
100 alerts per seed. Removals per seed: mean 39 (range 29-51). On average 32 real removals are found, 68 are false alarms and 6
removals are missed.

| Measure (gradient boosting, 20 seeds) | Value |
|---|---|
| Recall@5%, mean +- std | 0.84 +- 0.08 (95% CI of the mean 0.81-0.88) |
| Precision (share of the 100 alerts that were real removals) | 0.32 +- 0.05 |
| PR-AUC | 0.70 +- 0.07 (random is about 0.02) |
| Seeds with recall >= 0.80 | 16 of 20 |

Precision is computed from each seed's recall and number of removals with 100 alerts per seed.

| Profile | Recall@5%, mean +- std | Min-max | Seeds >= 0.80 |
|---|---|---|---|
| Easy | 0.94 +- 0.04 | 0.84-1.00 | 20/20 |
| Realistic | 0.84 +- 0.08 | 0.61-0.94 | 16/20 |
| Hard | 0.66 +- 0.08 | 0.50-0.80 | 1/20 |

**Shipped dataset (seed 42)**: recall 0.88 (30 of 34 removals), bootstrap 95% CI 0.76-0.97; 100 alerts, 30 real, 70 false alarms,
4 removals missed. The target is usually met, not guaranteed: the Realistic worst seed is 0.61. On the Hard profile 35% of
removals give no sensor warning by design, so recall is capped near one minus that share; that is a ceiling set by the data, not a
modelling failure. If a third of removals show no precursor in the data we collect, no model can reach 80% recall at 5 alerts per
100 on that data.

## 10. Why a component is flagged (SHAP)

SHAP shows how much each of the 50 features pushed one component's score up or down. Input: that component's features for the
month, scored by the boosting model. The global chart (`reports/global_importance.png`) is the average size of the push over the
validation months; the top drivers are oil temperature vs similar components, cycles since last fault, distinct fault codes and
current draw. Each alert lists its top 3 reasons in plain words (one per signal type). The reasons show what the model relied
on, not confirmed causes.

Alert = a component in the top 5% by risk score that month (20 of 400). The outcome is judged over the next 30 flight cycles.
Risk score = predicted chance of removal within 30 cycles; rank = position among the 400 components scored that month. `std` =
standard deviations from the average of similar components of the same type (full tables in `reports/example_predictions.md`).

| Case | Component, month | Risk score (0-1) and rank of 400 | Top reasons (SHAP) | Outcome |
|---|---|---|---|---|
| True positive: flagged and removed | Hydraulic pump C00542, Sep 2023 | 0.99 (1st) | Pressure readings unstable (4.1 std below normal); oil temperature 1.7 std above similar components; 16 days since last maintenance | Removed within 30 cycles: correct alert |
| False positive: flagged, not removed in time | Bleed-air valve C00592, Nov 2023 | 0.88 (2nd) | Pressure unstable; oil temperature 3.4 std above similar components; current draw rising; 5 faults in the last 30 cycles | Removed about 1 month later: counted as a false alarm, but the warning came first |
| False negative: removed, not flagged | Hydraulic pump C00471, Dec 2023 | 0.003 (229th) | No strong signal: largest sensor deviation only 1.2 std; current draw slightly up; 79 days since last maintenance | Removed with no warning: a sudden failure no model can see |

## 11. Cold start, missing data and drift

| Situation | How the model handles it | What we tested (recall@5%) | How we monitor it |
|---|---|---|---|
| New components (cold start) | Clues compare a component with its type average, so it still gets a score from current readings; history features are missing and the row gets a low-confidence flag | New components fail early: 0.90. One type never seen: 0.65. All types never seen: 0.18 | Share of low-confidence components; new types: build baselines, retrain |
| Missing data | Missing sensor readings stay missing, with an is-missing flag per sensor (a stopped sensor is itself a warning sign); boosting handles gaps natively, logistic uses the median from training | 20% more missing readings: 0.76 (was 0.88) | Missing rate per sensor; a jump means a broken data feed: fix the feed |
| Model drift | Fixed training baselines; stress-tested on simulated shifts; retrain on a trigger, the new model must beat the current one and a person approves | Sensor offset or 30% more flying: 0.88. Failure rate doubled: 0.63 | PSI/KS on inputs per airline; alerts per 100 at the fixed cutoff; rolling recall below 60% for 3 months |

**PSI** (Population Stability Index) compares the share of rows per value range now vs in training; below 0.1 stable, above 0.25
major shift. **KS** (Kolmogorov-Smirnov) is the largest gap between the two cumulative distributions, from 0 (same) to 1. Both
need no removal labels. All results are on our own simulator, so robustness is shown only for the shifts we modelled.

**Robustness to data drift**: the test months are re-generated with one change and scored by the already trained model
(`reports/stress_test.json`).

| What changed | Drift type | Removals caught | Alerts per 100 at the fixed cutoff | How we would detect it |
|---|---|---|---|---|
| Nothing (baseline) | - | 88% | 4.0 | - |
| One airline's oil sensors read high | Covariate shift | 88% | 4.0 | PSI/KS per airline |
| 30% more flying | Covariate shift | 88% | 4.0 | PSI/KS on flight hours |
| 20% more missing readings | Data-quality shift | 76% | 4.9 | Missing rate per sensor |
| Failures twice as frequent (78 removals) | Prior (base-rate) shift | 63% | 8.3 | Alerts per 100 jump |
| New components fail early | Concept drift | 90% | 4.6 | Rolling recall; low-confidence flag |
| One component type never seen | Unseen category | 65% | 3.0 | Low-confidence flag |
| All component types never seen | Unseen category | 18% | 0.1 | Alerts per 100 collapse |

Robust to covariate shift; weak on unseen component types and when the failure rate doubles. Normal rule: always 5 per 100. Fixed
cutoff (score 0.035, set on validation): alert volume moves with the data, so it signals drift.

## 12. Production architecture (proposed)

Adapted from an MLOps reference architecture for a monthly batch job on CPU. No GPU serving is needed; with much larger volume and
a GPU node, XGBoost on GPU is an option (not tested here).

- **Training and evaluation (CI runner)**: labelled monthly snapshots from the MRO systems, schema-validated -> feature
  engineering, logistic regression and gradient boosting, calibration, time-based split -> evaluate (recall@5% alerts, PR-AUC,
  bootstrap 95% CI) and pass a quality gate.
- **GitHub**: repository (code, data schema, pinned model tag) and GitHub Actions (CI/CD: tests, training, release). Actions starts
  the pipeline; a retrain trigger from monitoring comes back to it and a person approves.
- **Cloud registry and storage**: MLflow (experiment tracking and model registry) logs runs; the approved model version is released
  as a container image in the registry (Docker image, tag = model version). Only the approved model version is promoted.
- **Production, monthly batch scoring**: MRO systems (usage, sensor readings, fault codes) -> monthly scoring job (container on CPU:
  features and a risk score per component, `src/score.py`) -> maintenance planning (top-5% alert budget with SHAP reasons;
  engineers inspect and record findings, which become new labels).
- **Monitoring and feedback loop**: a monitoring job (data drift tests PSI/KS, alerts per 100, missing-value rate) feeds
  dashboards and alerts (data drift per airline, missing readings, alert volume); a performance check (recall, precision) uses
  observed removals 30 cycles later.

## 13. Monitoring and retraining

| What we watch | How | What it catches | Action |
|---|---|---|---|
| Inputs per airline | Compare with training data (PSI and KS: scores for how much data changed) | One airline's sensors drifting; hidden when the fleet is pooled | Check sensors and data feed |
| Missing readings | Share missing per sensor | A broken data feed | Fix the feed |
| Alerts per 100 | Count under a fixed score cutoff | A change in the failure rate | Review capacity if it moves by more than 2 |
| Low-confidence share | Share of components marked low-confidence | New component types | Build baselines, retrain |
| Recall and precision | Past alerts vs removals after 30 cycles | The model getting worse | Retrain if recall < 60% for 3 months |

- These numbers are starting points: even our unshifted test months show a high PSI on flight hours, so thresholds are tuned on
  normal month-to-month variation. PSI says an input changed, not that the model got worse; recall, measured once removals are
  observed, is the final check.
- Retrain every quarter or on a trigger: time split, the new model must beat the current one on the latest periods, a person
  approves; new component types need baselines first.
- Scoring needs each component's full history and the whole fleet for each period: a one-row file always raises one alert and gets
  `low` confidence.

## 14. Limitations and future work

| Limitation | Planned next step |
|---|---|
| Simulated, easy data: an untrained rule already reaches recall 0.74 | Validate on real maintenance data under a data-sharing agreement |
| Only 34 removals in the test months: a single recall estimate can be off by about 10 points | Re-run the evaluation on more months as they arrive (rolling backtests), keeping confidence intervals |
| Weak spots: unseen component types (recall 0.65) and a doubled failure rate (0.63) | Handle new component types better; automatic drift reports and safe model updates |
| No cost of an alert vs a missed failure; "30 cycles = 1 month" is a simplification | Set the alert threshold from real inspection and disruption costs; predict time to removal instead of a fixed 30-cycle window |
| No fault chapter codes, flight phase, operating conditions or "no fault found" labels | Add these features and labels to the schema |

Also worth stating: the main split is not aircraft-disjoint (aircraft-level leakage is checked separately, with small groups);
the tuning folds hold about two months of removals each and are noisy; real short-haul aircraft fly about 4 to 8 cycles a day, so a
real deployment would score more often and map the window to the check schedule; explanations describe the model's reasoning, not
causes.
