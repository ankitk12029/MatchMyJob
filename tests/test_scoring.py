import pytest

from scoring import conf_label, lexical_overlap_boost


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
        ("Software Developer", "Software Engineer", 1),   # "software"
        ("Registered Nurse", "Registered Nurse", 2),       # "registered", "nurse"
        ("Data Scientist", "Financial Analyst", 0),
    ],
)
def test_lexical_overlap_boost_is_003_per_shared_word(query_title, kb_title, shared_words):
    assert lexical_overlap_boost(query_title, kb_title) == pytest.approx(0.03 * shared_words)
