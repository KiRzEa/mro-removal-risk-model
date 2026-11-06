import pytest

from src.generate_mock_data import generate
from src.schema import TIME_COLUMN
from src.split import aircraft_folds, split_by_period


@pytest.fixture(scope="module")
def snapshots():
    return generate(seed=5, difficulty_name="realistic", aircraft_count=20)


def period_count(frame):
    return frame[TIME_COLUMN].nunique()


def test_default_split_has_15_5_and_5_periods(snapshots):
    split = split_by_period(snapshots)
    assert [period_count(part) for part in split] == [15, 5, 5]


def test_periods_are_in_time_order_without_overlap(snapshots):
    train, validation, test = split_by_period(snapshots)
    assert train[TIME_COLUMN].max() < validation[TIME_COLUMN].min()
    assert validation[TIME_COLUMN].max() < test[TIME_COLUMN].min()
    assert len(train) + len(validation) + len(test) == len(snapshots)


def test_boundaries_can_be_moved_by_one_period(snapshots):
    earlier = split_by_period(snapshots, train_count=14, validation_count=5, test_count=5)
    assert [period_count(part) for part in earlier] == [14, 5, 5]


def test_too_few_periods_is_an_error(snapshots):
    short = snapshots[snapshots[TIME_COLUMN] < snapshots[TIME_COLUMN].sort_values().unique()[10]]
    with pytest.raises(ValueError, match="periods"):
        split_by_period(short)


def test_aircraft_folds_cover_every_aircraft_exactly_once(snapshots):
    folds = aircraft_folds(snapshots, fold_count=5, seed=1)
    everyone = [aircraft for fold in folds for aircraft in fold]
    assert sorted(everyone) == sorted(snapshots["aircraft_id"].unique())
    assert len(everyone) == len(set(everyone))
    assert all(len(fold) == 4 for fold in folds)


def test_aircraft_folds_are_deterministic_for_a_seed(snapshots):
    first = aircraft_folds(snapshots, 5, seed=1)
    assert first == aircraft_folds(snapshots, 5, seed=1)
    assert first != aircraft_folds(snapshots, 5, seed=2)
