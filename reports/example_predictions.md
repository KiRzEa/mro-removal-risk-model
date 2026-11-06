# Example predictions and explanations

Scores are calibrated risk estimates; reasons come from the gradient boosting model. All data is synthetic.

Reasons are signals the model relied on, not confirmed causes: recent maintenance, for example, may follow a problem instead of causing it. In the history tables a value of +2 means two standard deviations above the average for the same component type, and a negative value means below it.

## True positive: flagged and removed

Component C00542, period 2023-09-23: risk score 0.987, rank 1 in its period, flagged.

Top reasons:

1. Pressure readings are more unstable than usual
2. Oil temperature is well above the level of similar components
3. 16 days since the last maintenance action

Last four snapshots (sensor values as standard deviations from similar components):

| period | vibration z | oil z | pressure z | current z | faults | last code |
|---|---|---|---|---|---|---|
| 2023-06-25 | 0.0 | 3.7 | -2.2 | 3.3 | 1 | F520 |
| 2023-07-25 | 1.5 | 2.4 | -2.0 | 3.3 | 1 | F100 |
| 2023-08-24 | missing | 2.5 | -3.6 | 2.9 | 5 | F520 |
| 2023-09-23 | missing | 1.7 | -4.1 | 1.6 | 5 | F415 |

The model ranked this component near the top of its period and it was removed within the next 30 cycles, as predicted. The reasons match the history below.

## False positive: flagged but not removed in time

Component C00592, period 2023-11-22: risk score 0.884, rank 2 in its period, flagged.

Top reasons:

1. Pressure readings are more unstable than usual
2. Oil temperature is well above the level of similar components
3. Current draw rose since the previous snapshot

Last four snapshots (sensor values as standard deviations from similar components):

| period | vibration z | oil z | pressure z | current z | faults | last code |
|---|---|---|---|---|---|---|
| 2023-08-24 | missing | missing | -2.8 | 0.8 | 2 | F100 |
| 2023-09-23 | 1.7 | 2.3 | -2.0 | 1.7 | 3 | F520 |
| 2023-10-23 | 1.7 | missing | -1.9 | 1.4 | 2 | F415 |
| 2023-11-22 | missing | 3.4 | -2.7 | 1.5 | 5 | F520 |

This alert was not followed by a removal in the next 30 cycles, but the part was removed about 1 period later. By the strict 30-cycle label this is a false alert, but the warning came before the removal. Such alerts are the cost of the 5% budget.

## Missed positive: removed but not flagged

Component C00471, period 2023-12-22: risk score 0.003, rank 229 in its period, not flagged.

Strongest risk signals (none strong enough to flag):

1. Current draw rose since the previous snapshot
2. Pressure level is unusual
3. 79 days since the last maintenance action

Last four snapshots (sensor values as standard deviations from similar components):

| period | vibration z | oil z | pressure z | current z | faults | last code |
|---|---|---|---|---|---|---|
| 2023-09-23 | 0.3 | missing | 0.7 | -0.2 | 0 | F310 |
| 2023-10-23 | -0.3 | 1.6 | -0.1 | -0.6 | 0 | F310 |
| 2023-11-22 | 0.3 | -0.2 | -0.7 | -0.4 | 1 | F210 |
| 2023-12-22 | -1.0 | 0.0 | -1.2 | 0.2 | 0 | F210 |

The model did not flag this removal. Its largest sensor deviation was 1.2 standard deviations from similar components, so it looked close to normal; removals without a visible warning cap how much any model can catch.
