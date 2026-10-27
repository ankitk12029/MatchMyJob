import numpy as np
import pandas as pd
import pytest

from scoring import (
    BM25_WEIGHT, TITLE_BOOST_WEIGHT, build_bm25_index, conf_label,
    lexical_overlap_boost, lexical_rerank, title_words,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        (65, "High confidence"),
        (64.9, "Moderate confidence"),
        (50, "Moderate confidence"),
        (49.9, "Low confidence"),
    ],
)
def test_conf_label_boundaries(value, expected):
    assert conf_label(value) == expected


@pytest.mark.parametrize(
    "query_title, kb_title, shared_words",
    [
        ("Software Developer", "Software Engineer", 1),    # "software"
        ("Registered Nurse", "Registered Nurse", 2),        # "registered", "nurse"
        ("Registered Nurse", "Registered Nurses", 2),       # plural is stemmed
        ("Data Scientist", "Financial Analyst", 0),
        ("Director of Sales", "Sales Director", 2),         # stop word "of" ignored
    ],
)
def test_lexical_overlap_boost_per_shared_stemmed_word(query_title, kb_title, shared_words):
    assert lexical_overlap_boost(query_title, kb_title) == pytest.approx(
        TITLE_BOOST_WEIGHT * shared_words)


def test_title_words_stems_and_drops_stop_words():
    assert title_words("Supervisors of Retail Analysts") == {"supervisor", "retail", "analyst"}


def _toy_kb():
    return pd.DataFrame({
        "Title": ["Retail Salespersons", "Software Developers", "Registered Nurses"],
        "All_Alt_Titles": ["Cashier, Sales Clerk", "Programmer", "RN"],
        "Description": ["Sell merchandise in a store", "Write software", "Provide patient care"],
    })


def test_lexical_rerank_adds_bm25_and_title_terms():
    kb = _toy_kb()
    bm25 = build_bm25_index(kb)
    cosine = np.zeros((1, 3), dtype=np.float32)
    out = lexical_rerank(cosine, ["retail sales clerk"], ["Retail Sales Clerk"], kb, bm25)
    assert out.shape == (1, 3)
    assert int(np.argmax(out[0])) == 0                       # Retail Salespersons wins
    assert out[0].max() <= BM25_WEIGHT + 2 * TITLE_BOOST_WEIGHT + 1e-6
    assert out[0, 2] == pytest.approx(0.0)                   # no keyword or title overlap


def test_lexical_rerank_empty_title_adds_only_bm25():
    kb = _toy_kb()
    bm25 = build_bm25_index(kb)
    cosine = np.zeros((1, 3), dtype=np.float32)
    out = lexical_rerank(cosine, ["write software"], [""], kb, bm25)
    assert out.max() <= BM25_WEIGHT + 1e-6
