import re

import pandas as pd
import pytest

from config import KB_PATH

EXPECTED_COLUMNS = {
    "O*NET-SOC Code",
    "Title",
    "Description",
    "Structured_Tasks",
    "All_Alt_Titles",
    "All_Tech_Skills",
    "All_Tools",
}

SOC_CODE_RE = re.compile(r"^\d{2}-\d{4}\.\d{2}$")


@pytest.fixture(scope="module")
def kb_df():
    assert KB_PATH.exists(), f"missing {KB_PATH}"
    return pd.read_csv(KB_PATH)


def test_kb_has_1016_rows(kb_df):
    assert len(kb_df) == 1016


def test_kb_has_expected_columns(kb_df):
    assert set(kb_df.columns) == EXPECTED_COLUMNS


def test_kb_no_nulls_in_soc_code_or_title(kb_df):
    assert kb_df["O*NET-SOC Code"].isnull().sum() == 0
    assert kb_df["Title"].isnull().sum() == 0


def test_kb_soc_codes_match_expected_format(kb_df):
    bad = kb_df.loc[~kb_df["O*NET-SOC Code"].astype(str).str.match(SOC_CODE_RE)]
    assert bad.empty, f"{len(bad)} SOC codes do not match {SOC_CODE_RE.pattern}"
