# Component removal risk model: approach, results and design decisions

All figures come from synthetic data produced by the project's own generator. Numbers below are copied from
`reports/*.json`; regenerate them with the commands in `README.md`.

## 1. Summary

The model ranks installed components by their risk of an unscheduled removal within the next 30 flight cycles and
flags the top 5% of each monthly period. A gradient boosting model, chosen on validation data over a logistic
regression baseline, catches **0.88** of the 34 removals in the final test window at that budget
(95% interval 0.76 to 0.97): the point estimate is above the 80% target, but the interval's lower end is below it.
Because one draw is not a fair summary of a random generator, the whole pipeline was repeated on 20 regenerated
datasets: mean recall 0.84 (95% interval for the mean 0.81 to 0.88; range over the 20 datasets 0.61 to 0.94), and 80% was
reached on 16 of 20 (95% interval 58% to 92%). So the target is usually met, not guaranteed.

Two honest framings belong next to those numbers. First, the data is synthetic and easy by construction: the
generator makes the within-period spread of the sensors an almost direct function of how close a removal is, so a
fixed rule that simply sums the four spread deviations (no training) reaches 0.74 on average, the same as
logistic regression (0.74). Boosting adds about 0.11 over the best simple rule, consistently across seeds. What
the study shows is that the pipeline recovers a planted signal without leakage and degrades in a sensible way under
the modelled shifts, not that it would predict removals on a real fleet. Second, on the committed seed-42 draw that same
spread rule happens to score 0.91, above the model's 0.88; the seed study is the fair comparison.

## 2. Data

`data/mock_components.csv` has 10,000 rows: one snapshot per installed component every
30 flight cycles, 25 monthly periods, 400 component positions on 80 aircraft, five component types, about 2%
positives. Columns (`data/SCHEMA.md`) cover aircraft and part attributes, usage (cycles and hours since new,
since overhaul), four sensors (mean and spread over the period, can be missing), fault messages, maintenance
history and the label. The schema follows how public maintenance data is organised: machine attributes, telemetry,
error log, maintenance records and failure records, flattened into one feature snapshot per component and period.
The generator has easy, realistic and hard profiles; the realistic profile (15% of removals give no sensor
warning) was fixed before any model was run and is used for the headline.

What exploratory analysis of the data found, and what it changed:

- Removal rates are similar across types and operators (1.8% to 2.3%); per-type counts are too small to report.
- Sensors drift away from their type average over about 8 to 10 periods before a removal; faults rise from 0.3 to 3.3
  per period and high-risk codes from 9% to 63%. Age matters (no removals in the youngest decile, 6.3% in the oldest);
  time since overhaul does not.
- Deviation from the type average (AUC 0.90) and the within-period spread (0.89 to 0.92) separate removals far better
  than raw means (0.65 to 0.73) or the change since the last snapshot (0.56 to 0.58): features follow that order.
- Vibration is missing 25% of the time just before a removal against 8% otherwise: missing indicators are kept.
- Extreme values are mostly real degradation (21% of values beyond 4 standard deviations are positives), so raw
  clipping was rejected.
- `aircraft_age_years` trends with the calendar (PSI 0.24) and has no causal role, so it is not a feature.

## 3. Data preparation

- **Missing values**: about 8% of sensor values are missing. Every sensor has a missing indicator. The boosting model
  reads missing values natively; the logistic model fills with the training median. Rows with every sensor missing are
  still scored and flagged low-confidence.
- **Outliers**: the logistic model clips type-deviation scores at 5 standard deviations (45 values in training); the
  boosting model sees raw values, since trees are robust to extremes and an extreme is often the failure signal.
- **Categoricals**: one-hot for the logistic model (unknown values ignored); ordinal codes for boosting (unknown
  values become missing). Identifiers and dates are never features.
- **Class imbalance**: 2.1% positives in training. Both models use balanced class weights
  (24.19:0.51). SMOTE was not used: it would invent trajectories that do not exist and complicate the time order. Ranking
  metrics, not accuracy, are reported.

## 4. Splitting and leakage prevention

- **Time**: periods 0 to 14 train (124 removals), 15 to 19 validate (45), 20 to 24 test (34). The test window also holds
  the usage drift and the replacement components, as a deployment would.
- **Look-ahead**: a feature reads only the row and earlier rows of the same component. A test changes every later row
  and checks that earlier features do not move.
- **Preprocessing and baselines**: type baselines, imputers, scalers and encoders are fitted on training rows only.
- **Choices**: settings are tuned on forward-chaining folds inside training (always validating on later periods); the
  model is chosen, calibrated and the fixed cutoff set on validation; the test window picks nothing.
- **Aircraft and component level**: a stricter check combines both. Aircraft are split into 5 groups; for each group the models
  are trained on the earlier periods of the other 80% of aircraft and scored on the test window of the group they were not
  trained on, and the scores of all groups are pooled and ranked fleet-wide in each period, as in the main evaluation. Recall at the 5%
  budget is 0.94 for gradient boosting (95% interval 0.84 to 1.00; range over 5 random groupings 0.91 to 0.97)
  and 0.88 for logistic regression (range 0.82 to 0.91), on the same 34 test removals. The
  five groupings reuse those removals, so they are not independent evidence, and the settings were tuned beforehand on all
  aircraft, so this checks the fitted models, not the tuning. Held-out aircraft keep their own earlier history in the
  features. The result says recall does not fall when the model is trained without an aircraft; it is not a test of new
  fleets.

## 5. Models and selection

Two families on identical features, splits and tuning effort: regularised logistic regression (interpretable
baseline) and histogram gradient boosting. Selection uses validation recall at the 5% budget, with average
precision as tie-breaker. Boosting was chosen on every one of the 20 seeds.

| Model | Recall at 5% (95% interval) | Precision | Lift | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| gradient boosting | 0.88 (0.76 to 0.97) | 0.30 | 17.6 | 0.77 | 0.97 |
| logistic regression | 0.91 (0.80 to 1.00) | 0.31 | 18.2 | 0.52 | 0.98 |

Test window, 34 removals. The intervals overlap, so this single window cannot separate the models; the
seed study can: gradient boosting averages 0.84 against 0.74 for logistic regression, which reaches 80% on only
25% of datasets. Boosting also has the clearly better PR-AUC (0.77 against 0.52).

## 6. Evaluation and threshold selection

**Impact framing.** Five alerts per 100 components is the inspection capacity given in the brief. A missed removal
costs an unscheduled event; an extra alert costs an inspection. The recall-versus-alerts curve
(`reports/recall_vs_alerts.png`) shows the trade-off: the smallest budget that reaches 80% recall is
4 per 100 on validation and 3 per 100 on test for gradient boosting (7 per 100 and 5 per 100 for logistic regression).

**Alert rule.** Two ways to turn scores into alerts are compared. The operating rule ranks components within each monthly
period and flags the top 5%, so the capacity limit holds by construction and the target is evaluated this way. The
alternative is one fixed cutoff on the calibrated score, set on validation (the 95th percentile of validation scores, 0.035)
and applied to every later month.

| Policy | Test recall | Alerts per 100 | Stress: doubled failure rate | Stress: all component types unseen |
|---|---|---|---|---|
| Top 5% per month | 0.88 | 5.0 exactly | recall 0.63 at 5.0 alerts | recall 0.18 at 5.0 alerts |
| Fixed validation cutoff | 0.88 | 3.95 on average (5.3, 3.8, 5.0, 2.3, 3.5 by month) | recall 0.86 at 8.3 alerts | recall 0.00 at 0.1 alerts |

The fixed cutoff does not hold the budget, so it is not the operating rule, but its alert rate moves when the fleet changes,
which makes it the drift signal in the monitoring plan: a capped rule always shows 5 alerts and hides the change. The
doubled-failure case also shows the limit of a fixed budget, since capacity has to grow with the failure rate.

**Calibration.** A sigmoid calibrator fitted on validation; Brier score of the chosen model on test 0.0067 before and 0.0063 after (compare models only after calibrating both; raw scores from class-weighted models are not probabilities).
Ranking, and therefore the alert list, is unchanged by calibration.

**Simple references** (no training, test window): ranking by age alone catches 0.18, by the latest
vibration deviation 0.62, at random 0.03, and by the sum of the four sensor-spread
deviations 0.91. The last rule was fixed in advance, not picked on test results; it is a strong
baseline because spread is the dominant signal in this generator. In the seed study the "best simple rule" is the best
of the four on each seed, which favours the baseline.

**What differs between the profiles** (`DIFFICULTIES` in `src/generate_mock_data.py`; every profile keeps about 2% positives
and the same 80 aircraft, 5 slots and 25 periods):

| Setting | Easy | Realistic | Hard | Effect |
|---|---|---|---|---|
| Removals with no sensor warning | 5% | 15% | 35% | Cannot be predicted from sensors at all |
| Strength of the degradation signal | 4.0 | 3.0 | 2.0 | How far sensors drift before a removal |
| Sensor noise scale | 0.8 | 1.0 | 1.3 | Rescales every sensor quantity (noise, offsets, drift, spikes) together, so it does not change the standardised features or the labels (verified); only raw sensor units differ |
| Random missing sensor values | 2% | 6% | 15% | Measured share of missing values: 3.9%, 8.0%, 16.8% |
| Removals whose label is dropped (label noise) | 0% | 1.5% | 4% | A real removal recorded as no removal |
| Late-period drift | none | mild (0.3) | strong (0.8) | Sensor offset and higher usage in the last 5 periods |

Shared by all profiles: rare one-off sensor spikes (0.5%) and sensor dropouts (0.4%), a small share of scheduled removals,
vibration readings that go missing before a removal for 20% of components, and the same fault-code and maintenance logic.

**Three difficulty profiles.** The generator has easy, realistic and hard settings (the share of removals that give no sensor
warning and the strength of the degradation signal). Each profile was run once on its own seed-42 dataset and as a full
20-seed study with the whole pipeline repeated (`reports/stability*.json`):

| Profile | Test removals (seed 42) | Seed 42 recall (95% interval) | 20 seeds: mean (2.5th to 97.5th percentile) | Seeds reaching 0.80 |
|---|---|---|---|---|
| easy | 41 | 0.95 (0.88 to 1.00) | 0.94 (0.84 to 0.99) | 100% |
| realistic | 34 | 0.88 (0.76 to 0.97) | 0.84 (0.68 to 0.94) | 80% |
| hard | 31 | 0.68 (0.50 to 0.83) | 0.66 (0.51 to 0.79) | 5% |

The target is met comfortably on easy data, usually on realistic data and almost never on hard data. The model beats the best
simple rule by 0.08 to 0.11 on every profile. The hard profile shows where the approach stops working: in this generator,
removals with no sensor warning are unpredictable by design.

**Why the hard profile cannot reach the target** (`python -m src.hard_profile_study`, 8 datasets, untuned, test window). The gap is a
ceiling set by the generator, not a modelling failure: removals with no precursor carry no signal by construction, so this
study confirms the model reaches that ceiling, not that real fleets behave this way. Changing one hard setting at a time to its realistic level:

| Hard profile with one change | Mean recall at 5 alerts per 100 |
|---|---|
| As defined (35% sudden failures) | 0.68 |
| No sudden failures | 1.00 |
| Sudden failures at the realistic level (15%) | 0.85 |
| Missing values at the realistic level (6%) | 0.69 |
| No label noise | 0.67 |
| Degradation signal at the realistic level | 0.67 |

Removals with no sensor warning are almost the whole gap: recall is close to one minus the share of sudden failures, and
the other settings move it by about 0.01. With more inspection capacity the target is within reach but expensive:

| Alerts per 100 | 5 | 8 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| Hard profile recall | 0.68 | 0.73 | 0.76 | 0.79 | 0.85 |

What to tell the company: if a third of removals show no precursor in the data we collect, no model can reach 80% recall
at 5 alerts per 100 on that data. Options, in order of promise: (1) feed the model more data that could warn of sudden
failures, such as operating conditions, inspection or shop findings and pilot reports; (2) raise the alert capacity, about 15
to 20 alerts per 100 on this profile; (3) report warned and unwarned removals separately and agree a lower target for the
unwarned ones; (4) cover age-related risk with preventive replacement. Tuning the model, resampling or lowering the cutoff
would not create information that is not in the data.

**What the alerts look like** (test window, top 5% per month, `alert_outcomes_top_5_percent` in `reports/metrics.json`):
100 alerts were raised for 34 removals: 30 true positives, 70 false positives, 4 missed removals. Per month, recall was
8 of 8, 7 of 9, 6 of 6, 2 of 3 and 7 of 8. Some false alerts are near-misses, because the label is a hard 30-cycle window:
for the first four months, 17 of the 57 false alerts were followed by a removal in cycles 31 to 60, and counting those,
precision within 60 cycles is 0.50 against 0.30 for the strict 30-cycle label.

**Stability across 20 regenerated datasets** (realistic profile, whole pipeline repeated, `reports/stability.png`). With 20
values, the 2.5th and 97.5th percentiles are essentially the minimum and maximum, so the range is shown instead:

| | Mean (95% interval for the mean) | Range over 20 datasets | Datasets at 0.80 or more (Wilson interval) |
|---|---|---|---|
| gradient boosting (chosen) | 0.84 (0.81 to 0.88) | 0.61 to 0.94 | 16 of 20 (95% interval 58% to 92%) |
| logistic regression | 0.74 (0.70 to 0.78) | 0.59 to 0.94 | 5 of 20 (95% interval 11% to 47%) |
| sum of spread deviations (no training) | 0.74 (0.70 to 0.77) | 0.61 to 0.90 | 4 of 20 (95% interval 8% to 42%) |
| best simple rule per dataset | 0.74 (0.70 to 0.77) | 0.61 to 0.90 | 4 of 20 (95% interval 8% to 42%) |

**Model comparisons** (paired over the 20 datasets, because both models see the same data; `p_values` in `reports/stability*.json`).
Boosting is above logistic regression on every profile (one-sided Wilcoxon signed-rank, p < 0.001), and above the best
simple rule on every profile (p < 0.001). The smallest possible p-value with 20 seeds is about 1e-6, and the author
chooses the number of seeds, so these say the gap is not luck across simulated datasets, nothing about real fleets. No
significance test is used for the 80% target; the claim is the share of datasets reaching it, with its interval above.

**Fair comparison protocol.** Same features and splits for both models and a small tuning grid for each (3 settings for logistic
regression, 4 for boosting), so the tuning effort is comparable in size, not identical; the test window picks nothing;
the realistic setting of 15% sudden failures was chosen before any model was run (stated by the author, not verifiable
from the repository); misses are reported as plainly as hits. The logistic model gets no interactions, so 0.74 is a
simple baseline, not a tuned linear ceiling. Two kinds of
uncertainty are shown separately: the component bootstrap (sampling noise in one dataset, conditional on the trained model and
the fixed alert flags, so it leaves out training variance and is optimistic about repeatability) and the seed study
(randomness of the simulation and of training).

**Target statement.** On the committed dataset the recall point estimate is 0.88, above the 80% target, with a
95% interval of 0.76 to 0.97 that reaches below it. Across regenerated datasets the target is reached on
16 of 20 (95% interval 58% to 92%), with a low of 0.61. On the hard profile it is not reached.

## 7. Explanations

Global drivers (`reports/global_importance.png`) come from SHAP values of the boosting model on a validation sample.
For a flagged component the three strongest signals of different kinds are turned into plain sentences, for example
"Oil temperature is well above the level of similar components" or "Pressure readings are more unstable than
usual". `reports/example_predictions.md` explains a true positive, a false positive (a different component, removed
one period later) and a missed positive (sensors looked normal, which no model can catch).

## 8. New components, missing data and drift

- **New components**: a component with fewer than 2 earlier snapshots is scored from its type baseline and current
  values and flagged low-confidence. In the base data new components never fail, so a stress test adds early
  failures: recall 0.90.
- **A component type the model has not seen** is the weak spot: recall falls to 0.65 when one of five types is unknown and to 0.18
  when all are, because the model relies on type baselines. Rows are flagged low-confidence; the operational answer is
  to compute baselines for a new type from its first two or three months and retrain.
- **Missing data**: scoring never fails on missing sensors. With 20% extra missing values from the drift window on,
  recall at the budget is 0.76 (0.62 to 0.90).
- **Drift**: stress tests regenerate the drift window with shifted settings and score it with the model trained on
  the unshifted data (`reports/stress_summary.png`):

| Scenario | Removals | Recall (95% interval) | Alerts per 100 with the fixed cutoff |
|---|---|---|---|
| no shift committed dataset | 34 | 0.88 (0.76 to 0.97) | 4.0 |
| sensor offset one operator | 34 | 0.88 (0.76 to 0.97) | 4.0 |
| higher usage | 34 | 0.88 (0.76 to 0.97) | 4.0 |
| more missing data | 34 | 0.76 (0.62 to 0.90) | 4.9 |
| low base rate | 17 | 1.00 (1.00 to 1.00) | 3.3 |
| high base rate | 78 | 0.63 (0.52 to 0.73) | 8.3 |
| faster degradation | 34 | 0.91 (0.80 to 1.00) | 3.1 |
| early failures of new components | 42 | 0.90 (0.80 to 0.98) | 4.6 |
| fresh seed 101 | 43 | 0.79 (0.67 to 0.90) | 5.1 |
| fresh seed 102 | 40 | 0.90 (0.80 to 0.98) | 5.0 |
| fresh seed 103 | 38 | 0.92 (0.83 to 1.00) | 4.9 |
| fresh seed 104 | 42 | 0.88 (0.77 to 0.98) | 4.7 |
| fresh seed 105 | 40 | 0.78 (0.63 to 0.90) | 4.8 |
| one unseen component type | 34 | 0.65 (0.48 to 0.80) | 3.0 |
| all component types unseen | 34 | 0.18 (0.06 to 0.31) | 0.1 |

  A one-operator sensor offset and a 30% usage increase barely move recall (0.88). A doubled removal rate
  (0.63) shows a hard limit: with 5 alerts per 100 and 4% of components failing, 80% recall needs a precision of
  about 60%, so capacity must grow with the failure rate. The fixed cutoff raises 8.3 alerts per 100 in that case,
  which is exactly the drift signal a capacity rule would hide. Fresh seeds with unchanged settings give 0.78 to 0.92.

**Which monitor catches which shift** (PSI = population stability index on the features, computed after dropping features that
grow with the calendar by construction, such as history length; PSI above 0.25 is conventionally a large shift):

| Shift | Pooled PSI of the affected feature | Per-operator PSI | Fixed-cutoff alerts per 100 | What catches it |
|---|---|---|---|---|
| Oil temperature offset on one operator | 0.14 (below 0.25, missed) | 2.6 for OP_C, about 0.0 for the others | 4.0 (unchanged) | PSI per operator |
| 30% higher usage | 6.7 (flight hours) | 6.5 to 7.2 in every operator | 4.0 | PSI on flight hours |
| 20% more missing sensor values | not in the top features | not in the top features | 4.9 | Share of missing values per sensor |
| Doubled failure rate | no feature shifts | no feature shifts | 8.3 | Alert rate under the fixed cutoff, then delayed-label recall |
| All component types unseen | no feature shifts | no feature shifts | 0.1 | Alert rate collapses; share of low-confidence rows |

No single monitor covers every case: input PSI misses changes in the failure rate, and the alert rate misses a sensor
offset. Two cautions: the committed test window already carries mild built-in usage drift, so flight hours show PSI 0.94
with no stress test at all; alert thresholds therefore have to be set against a baseline of normal month-to-month
variation, not copied from the conventional 0.25. And PSI says an input changed, not that the model got worse; recall,
measured once removals are observed 30 cycles later, is the final check.

## 9. Deployment, monitoring and retraining

- **Scoring**: a monthly (or after each flight batch) job builds the snapshot table from the MRO systems (aircraft and
  part master, sensor telemetry, fault messages, maintenance records), runs `src.score`, and writes score, rank,
  confidence and reasons to the work-planning board; the top 5% become inspection tasks.
- **Integration with maintenance planning**: the alert list drives inspection capacity; engineers confirm or dismiss each
  alert and record the shop finding (confirmed, no fault found). Those outcomes become labels, and the split between
  confirmed and unconfirmed removals can be used to refine the target.
- **Monitoring**: per-feature PSI and KS between training and the latest period, computed fleet-wide and per operator
  (a one-operator shift barely moves the fleet-wide value); the alert rate under a fixed cutoff; the share of missing
  sensor values per sensor; the share of low-confidence rows; and, once removals are observed 30 cycles later, recall at
  the budget and precision of past alerts. Rule of thumb: PSI above 0.25 on a main driver, an alert-rate drift beyond
  2 per 100, or recall falling below 0.6 on a rolling 3-month window triggers review.
- **Retraining**: on a schedule (quarterly) and when drift triggers fire, always with a time split and a champion versus
  challenger comparison on the latest periods before promotion; new component types need baselines first.

## 10. Design decisions

- Time split with a one-period label horizon, no embargo needed; test window never used for a choice.
- Rank-based alert budget per period instead of one probability threshold; fixed cutoff reported for drift.
- Class weights instead of oversampling; recall at the budget, PR-AUC, lift, calibration instead of accuracy.
- Type-deviation and spread features over change features, after the EDA; clipping applied only to the logistic model.
- Aircraft age excluded as a calendar-trending feature; prior removals kept and monitored.
- Headline claim is the multi-seed range, with the component bootstrap interval for any single window.
- The realistic profile was set to 15% removals without sensor warning before any model ran; it was not changed afterwards.

## 11. Limitations

- Synthetic data: results show the method, not a real fleet, and robustness covers only modelled shifts.
- The test window holds about 34 removals; one recall figure has an interval of roughly plus or minus 0.1.
- The generator has no operating-condition columns, no time since installation, simple fault codes without ATA chapter
  or flight phase, no infant mortality in the base data, no link between overhaul timing and failure, and counts all
  unscheduled removals as one class (no confirmed versus no-fault-found split).
- Unseen component types degrade the model sharply; per-type results cannot be reported (3 to 11 removals per type in
  the test window).
- Independent audit (a separate reviewer re-ran the pipeline on a copy): no leakage found. Shuffling training labels gave
  chance-level test recall (0 to 0.09), deliberately peeking at later rows made the no-look-ahead test fail, and recall
  recomputed from `reports/test_scores.csv` with plain pandas matched (30 of 34).
- The task is easy by construction: the within-period spread of vibration and pressure is an almost noiseless function of
  proximity to removal in the generator, and it dominates permutation importance. Results depend on that synthetic channel.
- Near-miss: a false alert followed by a removal 31 to 60 cycles later is a lead-time hit under a hard 30-cycle label; the precision within 60 cycles of 0.50 is a rough post-hoc diagnostic on four months, not a result.
- "30 flight cycles = one monthly snapshot" is a simulation convenience. Real short-haul aircraft fly about 4 to 8 cycles a day and planners work in check and slot windows, so a real deployment would score more often and map the window to the check schedule.
- No cost model: the report gives the recall trade-off curve but does not price an alert against a missed removal; the 5 per 100 comes from the brief and counts alerted positions per period, not workshop capacity. No-fault-found removals (often a large share for avionics and pumps) would be a separate target in practice.
- The 80% target is "usually met" (16 of 20 seeds), not guaranteed; the committed seed-42 draw (0.88) is above the seed mean (0.84).
- On the committed split logistic regression scored slightly higher on test (0.91 against 0.88) and the choice on validation was marginal; boosting wins on PR-AUC and on
  the paired seed comparison (18 wins, 0 losses), which is the evidence to cite.
- The main split is not aircraft-disjoint; aircraft-level leakage is checked separately with the unseen-aircraft test above, whose groups are small.
- Scoring needs the component's full history and the whole fleet for each period: a one-row file always raises one alert and
  gets `low` confidence, and scoring only the latest period drops the history features.
- The tuning folds hold about two months of removals each and are noisy.
- Explanations come from the boosting model and describe the model's reasoning, not causes.

## 12. Future work

Operating-condition and ATA-coded fault features; baselines that adapt for new component types; survival or
time-to-event modelling instead of a fixed 30-cycle window; cost-sensitive thresholds from real inspection and
disruption costs; a confirmed versus no-fault-found target; automated drift reports and champion versus challenger
retraining; validation on real MRO data under a data-sharing agreement.

## Sources for the schema comparison

NASA C-MAPSS turbofan data layout, the Azure predictive-maintenance sample (machines, telemetry, errors,
maintenance, failures), MRO time and cycle conventions (TSN, CSN, TSO, CSO) and removal record fields, and the NGAFID
and synthetic landing-gear datasets as the closest public data (schema comparison done during planning).
