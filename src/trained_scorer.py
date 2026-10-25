"""
Trained scorer: TF-IDF -> LSA (dimensionality reduction) -> supervised
classifier. This is a genuine *trained* model (unlike baseline_scorer.py's
untrained similarity threshold), and unlike embeddings_scorer.py it needs
no internet access to a model hub, so it's fully reproducible in any
locked-down environment -- including this one, which is why the results
below are real, not hypothetical.

Feature vector per (reference, student) pair, in the classic SBERT-style
"sentence pair" shape:
    [ref_lsa_vector ; student_lsa_vector ; |ref_lsa_vector - student_lsa_vector| ; cosine_similarity]

This is the same feature design sketched for the transformer upgrade in
embeddings_scorer.py -- swap TF-IDF+LSA for SBERT embeddings later and
the architecture barely changes, which is worth saying explicitly in
your methodology section.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess


class TrainedAnswerScorer:
    def __init__(self, n_components: int = 100, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer()
        self.svd = None
        self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._fitted = False

    def _lsa_vectors(self, texts: list[str]) -> np.ndarray:
        processed = [preprocess(t) for t in texts]
        tfidf = self.vectorizer.transform(processed)
        return self.svd.transform(tfidf)

    def _features(self, reference_answers: list[str], student_answers: list[str]) -> np.ndarray:
        ref_vecs = self._lsa_vectors(reference_answers)
        stu_vecs = self._lsa_vectors(student_answers)
        diff = np.abs(ref_vecs - stu_vecs)
        sims = np.array([
            cosine_similarity(ref_vecs[i:i + 1], stu_vecs[i:i + 1])[0][0]
            for i in range(len(reference_answers))
        ]).reshape(-1, 1)
        return np.hstack([ref_vecs, stu_vecs, diff, sims])

    def fit(self, train_df) -> "TrainedAnswerScorer":
        corpus = [preprocess(t) for t in list(train_df["reference_answer"]) + list(train_df["student_answer"])]
        self.vectorizer.fit(corpus)

        n_comp = min(self.n_components, len(corpus) - 1, self.vectorizer.transform(corpus[:1]).shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        self.svd.fit(self.vectorizer.transform(corpus))

        X = self._features(list(train_df["reference_answer"]), list(train_df["student_answer"]))
        y = train_df["human_score"].to_numpy()
        self.classifier.fit(X, y)
        self._fitted = True
        return self

    def score(self, reference_answer: str, student_answer: str, max_score: int = 2) -> dict:
        if not self._fitted:
            raise RuntimeError("Call .fit(train_df) before scoring.")
        X = self._features([reference_answer], [student_answer])
        predicted = int(self.classifier.predict(X)[0])
        proba = self.classifier.predict_proba(X)[0]
        confidence = float(np.max(proba))
        return {
            "predicted_score": predicted,
            "confidence": round(confidence, 4),
            "max_score": max_score,
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
    data_dir = Path(__file__).resolve().parent.parent / "data"
    train_df = pd.read_csv(data_dir / "real_answers_train.csv")
    test_df = pd.read_csv(data_dir / "real_answers_test.csv")

    scorer = TrainedAnswerScorer()
    scorer.fit(train_df)

    for _, row in test_df.head(5).iterrows():
        result = scorer.score(row["reference_answer"], row["student_answer"], row["max_score"])
        print(f"human={row['human_score']}  predicted={result['predicted_score']}  confidence={result['confidence']}")
