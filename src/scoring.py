"""
scoring.py — Pure scoring/labeling helpers shared by app.py, match_engine.py and match_engine_eval.py

Kept free of streamlit and torch (numpy only; rank_bm25 is imported lazily) so it can be imported and
unit-tested without launching the app or loading the model.
"""

import numpy as np

# Weights of the two lexical terms added to the weighted cosine score.
BM25_WEIGHT = 0.03            # normalized BM25 keyword score (tie-breaker)
TITLE_BOOST_WEIGHT = 0.05     # per shared stemmed word between query and KB title

_STOP_WORDS = {
    'and', 'or', 'the', 'of', 'a', 'in', 'for', 'to', 'at', 'by', 'an',
    'is', 'as', 'on', 'except', 'special', 'all', 'other', 'with', 'from',
}


def _stem(w: str) -> str:
    """Strip common English plural/agent suffixes for fuzzy title matching."""
    if len(w) > 5 and w.endswith('ers'):
        return w[:-2]   # supervisors -> supervisor, workers -> worker
    if len(w) > 3 and w.endswith('s'):
        return w[:-1]   # analysts -> analyst, nurses -> nurse
    return w


def title_words(title) -> set:
    """Stemmed, stop-word-filtered word set of a job title."""
    return {_stem(w) for w in str(title).lower().split()
            if w not in _STOP_WORDS and len(w) > 1}


def lexical_overlap_boost(query_title: str, kb_title: str) -> float:
    """Score increment from stemmed words shared between a query and KB title."""
    return TITLE_BOOST_WEIGHT * len(title_words(query_title) & title_words(kb_title))


def build_bm25_index(kb_df):
    """BM25 index over O*NET title + alternate titles + description."""
    from rank_bm25 import BM25Okapi
    corpus = []
    for _, row in kb_df.iterrows():
        doc = " ".join(filter(None, [
            str(row.get('Title', '')),
            str(row.get('All_Alt_Titles', ''))[:200],
            str(row.get('Description', ''))[:300],
        ]))
        corpus.append(doc.lower().split())
    return BM25Okapi(corpus)


def bm25_scores_for_query(bm25, query: str) -> np.ndarray:
    """BM25 scores of one query against every occupation, divided by their maximum."""
    scores = np.array(bm25.get_scores(query.lower().split()), dtype=np.float32)
    max_s = scores.max()
    if max_s > 0:
        scores = scores / max_s
    return scores


def lexical_rerank(cosine_scores, query_texts, query_titles, kb_df, bm25,
                   kb_title_words=None) -> np.ndarray:
    """Add the BM25 and title-word terms to weighted cosine scores.

    This is the single scoring path shared by the Streamlit app, the batch
    script (match_engine.py) and the evaluation (match_engine_eval.py):
    score = weighted cosine + BM25_WEIGHT * normalized BM25
            + TITLE_BOOST_WEIGHT * shared stemmed title words.

    cosine_scores: array (n_queries, n_kb). Returns a float32 array of the same shape.
    """
    cosine_scores = np.asarray(cosine_scores, dtype=np.float32)
    if kb_title_words is None:
        kb_title_words = [title_words(t) for t in kb_df['Title']]
    final = np.empty_like(cosine_scores)
    for i, (text, title) in enumerate(zip(query_texts, query_titles)):
        s = cosine_scores[i] + BM25_WEIGHT * bm25_scores_for_query(bm25, text)
        qw = title_words(title)
        if qw:
            for j, kw in enumerate(kb_title_words):
                n = len(qw & kw)
                if n:
                    s[j] += TITLE_BOOST_WEIGHT * n
        final[i] = s
    return final


def conf_color(v):
    if v >= 65: return "conf-high"
    if v >= 50: return "conf-mid"
    return "conf-low"


def conf_label(v):
    if v >= 65: return "High confidence"
    if v >= 50: return "Moderate confidence"
    return "Low confidence"
