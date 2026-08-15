import pandas as pd
import pytest

from config import PROCESSED_DATA_DIR

WEIGHTS_PATH = PROCESSED_DATA_DIR / "optimal_weights.csv"

EXPECTED_FIELDS = {"Tasks", "Description", "Skills", "Ofc_Title", "Alt_Titles", "Tools"}

# Published values from the paper — 4 decimal places.
EXPECTED_VALUES = {
    "Tasks": 0.2017,
    "Description": 0.1021,
    "Skills": 0.0063,
    "Ofc_Title": 0.3941,
    "Alt_Titles": 0.2898,
    "Tools": 0.0061,
}


@pytest.fixture(scope="module")
def weights_df():
    assert WEIGHTS_PATH.exists(), f"missing {WEIGHTS_PATH}"
    return pd.read_csv(WEIGHTS_PATH)


def test_weights_file_exists_and_parses(weights_df):
    assert not weights_df.empty
    assert set(weights_df.columns) == {"field", "weight"}


def test_weights_have_exactly_six_expected_fields(weights_df):
    assert set(weights_df["field"]) == EXPECTED_FIELDS


def test_weights_between_zero_and_one(weights_df):
    for w in weights_df["weight"]:
        assert 0 <= w <= 1


def test_weights_sum_to_one(weights_df):
    assert weights_df["weight"].sum() == pytest.approx(1.0, abs=0.001)


def test_weights_match_published_values(weights_df):
    by_field = weights_df.set_index("field")["weight"]
    for field, expected in EXPECTED_VALUES.items():
        assert round(float(by_field[field]), 4) == pytest.approx(expected, abs=0.0001)
