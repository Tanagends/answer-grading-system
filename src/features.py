"""
Domain-agnostic features for (model_answer, student_answer) pairs.

Why this exists: the earlier TF-IDF+LSA scorer learned the *vocabulary* of
one exam (Neural Networks), so it fell apart on any other subject. Every
feature here is defined *relative to the model answer* (overlap, coverage,
length ratio, typo-tolerant similarity), so the same trained classifier can
grade questions it has never seen -- which is what a real upload workflow
needs.

Features are computed per question group: the TF-IDF vocabulary/IDF is fit
on that question's model answer + the batch of student answers to it
(unsupervised, no labels needed), so it adapts to whatever subject the
instructor uploads.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess

FEATURE_NAMES = [
    "word_cos",       # TF-IDF cosine on stemmed words
    "char_cos",       # TF-IDF cosine on char n-grams (typo / spelling tolerant)
    "recall_w",       # IDF-weighted share of model-answer concepts the student covers
    "precision",      # share of student's content words found in the model answer
    "bigram_jacc",    # phrase-level overlap (stem bigrams)
    "len_ratio",      # log(student length / model length)
    "len_student",    # log(1 + student content words)
]


def _tokens(text: str) -> list[str]:
    return preprocess(text or "").split()


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


def _row_cos(m, i: int, j: int) -> float:
    a, b = m[i], m[j]
    if a.nnz == 0 or b.nnz == 0:
        return 0.0
    return float(cosine_similarity(a, b)[0][0])


def group_features(model_answer: str, student_answers: list[str]) -> np.ndarray:
    """Feature matrix (len(student_answers) x len(FEATURE_NAMES)) for ONE question."""
    ref_tok = _tokens(model_answer)
    stu_tok = [_tokens(s) for s in student_answers]

    word_docs = [" ".join(ref_tok)] + [" ".join(t) for t in stu_tok]
    char_docs = [(model_answer or "").lower()] + [(s or "").lower() for s in student_answers]

    try:
        wv = TfidfVectorizer(sublinear_tf=True).fit_transform(word_docs)
    except ValueError:  # empty vocabulary
        wv = None
    try:
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(char_docs)
    except ValueError:
        cv = None

    # IDF over this question's documents, used to weight concept recall
    idf = {}
    if wv is not None:
        vec = TfidfVectorizer(sublinear_tf=True).fit(word_docs)
        idf = dict(zip(vec.get_feature_names_out(), vec.idf_))

    ref_set = set(ref_tok)
    ref_bi = _bigrams(ref_tok)
    ref_weight = sum(idf.get(t, 1.0) for t in ref_set) or 1.0

    rows = []
    for k, toks in enumerate(stu_tok, start=1):
        s_set = set(toks)
        word_cos = _row_cos(wv, 0, k) if wv is not None else 0.0
        char_cos = _row_cos(cv, 0, k) if cv is not None else 0.0
        recall_w = sum(idf.get(t, 1.0) for t in ref_set & s_set) / ref_weight
        precision = len(ref_set & s_set) / len(s_set) if s_set else 0.0
        sb = _bigrams(toks)
        bigram = len(ref_bi & sb) / len(ref_bi | sb) if (ref_bi | sb) else 0.0
        len_ratio = float(np.clip(np.log((len(toks) + 1) / (len(ref_tok) + 1)), -3, 3))
        rows.append([word_cos, char_cos, recall_w, precision, bigram, len_ratio, np.log1p(len(toks))])
    return np.array(rows, dtype=float)


def build_features(df, ref_col="reference_answer", stu_col="student_answer", group_col="question_id") -> np.ndarray:
    """Feature matrix for a whole dataframe, computed question-by-question (row order preserved)."""
    X = np.zeros((len(df), len(FEATURE_NAMES)))
    pos = {idx: i for i, idx in enumerate(df.index)}
    for _, g in df.groupby(group_col, sort=False):
        feats = group_features(g[ref_col].iloc[0], g[stu_col].tolist())
        for row_i, idx in enumerate(g.index):
            X[pos[idx]] = feats[row_i]
    return X
