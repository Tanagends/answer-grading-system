"""
Baseline answer scorer: TF-IDF + cosine similarity.

This is intentionally the "boring but working" baseline every grading
system should be benchmarked against. It needs no pretrained model
downloads, so it runs anywhere. Report its correlation/kappa against
human scores in your results section, then show the transformer
upgrade (see embeddings_scorer.py) beats it -- that comparison is a
whole results section for the report, basically for free.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess


class BaselineAnswerScorer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self._fitted = False

    def fit(self, corpus: list[str]) -> "BaselineAnswerScorer":
        """Fit TF-IDF on a corpus of (preprocessed) reference + student answers."""
        processed = [preprocess(t) for t in corpus]
        self.vectorizer.fit(processed)
        self._fitted = True
        return self

    def similarity(self, reference_answer: str, student_answer: str) -> float:
        if not self._fitted:
            raise RuntimeError("Call .fit(corpus) before scoring.")
        ref = preprocess(reference_answer)
        stu = preprocess(student_answer)
        vecs = self.vectorizer.transform([ref, stu])
        sim = cosine_similarity(vecs[0], vecs[1])[0][0]
        return float(sim)

    def score(self, reference_answer: str, student_answer: str, max_score: int = 3) -> dict:
        """Map cosine similarity [0,1] to a predicted integer score [0, max_score]."""
        sim = self.similarity(reference_answer, student_answer)
        raw = sim * max_score
        predicted = max(0, min(max_score, round(raw)))
        return {
            "similarity": round(sim, 4),
            "predicted_score": predicted,
            "max_score": max_score,
        }


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("../data/sample_answers.csv")
    scorer = BaselineAnswerScorer()
    scorer.fit(list(df["reference_answer"]) + list(df["student_answer"]))

    for _, row in df.head(4).iterrows():
        result = scorer.score(row["reference_answer"], row["student_answer"], row["max_score"])
        print(f"[{row['question_id']}] human={row['human_score']}  predicted={result['predicted_score']}  sim={result['similarity']}")
