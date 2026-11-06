from typing import NamedTuple

import numpy as np
import pandas as pd

from src.config import TEST_PERIOD_COUNT, TRAIN_PERIOD_COUNT, VALIDATION_PERIOD_COUNT
from src.schema import TIME_COLUMN


class Split(NamedTuple):
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_by_period(
    snapshots: pd.DataFrame,
    train_count: int = TRAIN_PERIOD_COUNT,
    validation_count: int = VALIDATION_PERIOD_COUNT,
    test_count: int = TEST_PERIOD_COUNT,
) -> Split:
    dates = np.sort(snapshots[TIME_COLUMN].unique())
    needed = train_count + validation_count + test_count
    if len(dates) < needed:
        raise ValueError(f"Need {needed} snapshot periods, found {len(dates)}")

    validation_start = dates[train_count]
    test_start = dates[train_count + validation_count]
    test_end = dates[needed - 1]
    when = snapshots[TIME_COLUMN]
    return Split(
        train=snapshots[when < validation_start],
        validation=snapshots[(when >= validation_start) & (when < test_start)],
        test=snapshots[(when >= test_start) & (when <= test_end)],
    )


def aircraft_folds(snapshots: pd.DataFrame, fold_count: int, seed: int) -> list[set]:
    aircraft = np.sort(snapshots["aircraft_id"].unique())
    shuffled = np.random.default_rng(seed).permutation(aircraft)
    return [set(group) for group in np.array_split(shuffled, fold_count)]
