"""
Domain-agnostic scorer: relative-overlap features -> random forest.

Trained once on the ASAG data, saved to models/feature_scorer.joblib, then
reused to grade *any* uploaded question set (see batch_grader.py).

Output per answer: class 0 / 1 / 2 (incorrect / partial / correct) plus class
probabilities, so the batch grader can scale to any max marks and flag
low-confidence answers for human review.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from features import FEATURE_NAMES, build_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "feature_scorer.joblib"


class FeatureAnswerScorer:
    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
        )
        self.classes_ = np.array([0, 1, 2])

    def fit(self, df) -> "FeatureAnswerScorer":
        X = build_features(df)
        self.model.fit(X, df["human_score"].to_numpy())
        self.classes_ = self.model.classes_
        return self

    def predict_proba_df(self, df) -> np.ndarray:
        """Class probabilities (n x 3, columns = classes 0,1,2) for a dataframe with
        reference_answer / student_answer / question_id columns."""
        X = build_features(df)
        raw = self.model.predict_proba(X)
        out = np.zeros((len(df), 3))
        for col, cls in enumerate(self.model.classes_):
            out[:, int(cls)] = raw[:, col]
        return out

    def predict_df(self, df) -> np.ndarray:
        return self.predict_proba_df(df).argmax(axis=1)

    # Same interface as the other scorers, for single-pair use / eval scripts.
    def score(self, reference_answer: str, student_answer: str, max_score: int = 2) -> dict:
        import pandas as pd

        one = pd.DataFrame({"question_id": ["q"], "reference_answer": [reference_answer],
                            "student_answer": [student_answer]})
        proba = self.predict_proba_df(one)[0]
        cls = int(proba.argmax())
        return {"predicted_score": round(cls / 2 * max_score), "confidence": round(float(proba.max()), 4),
                "max_score": max_score}

    def save(self, path: Path = MODEL_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": FEATURE_NAMES}, path)

    @classmethod
    def load_or_train(cls, path: Path = MODEL_PATH) -> "FeatureAnswerScorer":
        """Load the saved model; if it's missing or unreadable (e.g. different scikit-learn
        version), retrain from data/real_answers_train.csv (takes a few seconds) and save it."""
        try:
            return cls.load(path)
        except Exception:
            import pandas as pd

            data = Path(__file__).resolve().parent.parent / "data" / "real_answers_train.csv"
            obj = cls().fit(pd.read_csv(data))
            try:
                obj.save(path)
            except OSError:
                pass
            return obj

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "FeatureAnswerScorer":
        if not Path(path).exists():
            raise FileNotFoundError(f"No trained model at {path} -- run `python src/train_feature_model.py` first.")
        blob = joblib.load(path)
        obj = cls()
        obj.model = blob["model"]
        obj.classes_ = obj.model.classes_
        return obj
