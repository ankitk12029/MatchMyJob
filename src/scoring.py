"""
scoring.py — Pure scoring/labeling helpers shared by app.py

Kept dependency-free (no streamlit, no torch) so it can be imported and
unit-tested without launching the app or loading the model.
"""

LEXICAL_OVERLAP_WEIGHT = 0.03


def lexical_overlap_boost(query_title: str, kb_title: str) -> float:
    """Score increment from words shared between a query and KB title."""
    query_words = set(str(query_title).lower().split())
    kb_words = set(str(kb_title).lower().split())
    overlap = len(query_words & kb_words)
    return LEXICAL_OVERLAP_WEIGHT * overlap


def conf_color(v):
    if v >= 72: return "conf-high"
    if v >= 58: return "conf-mid"
    return "conf-low"


def conf_label(v):
    if v >= 65: return "High confidence"
    if v >= 50: return "Moderate confidence"
    return "Low confidence"
