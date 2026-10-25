"""
Train the domain-agnostic scorer and report three evaluations.

  1. In-domain test      : held-out answers to questions seen in training
  2. Unseen-question CV  : 5-fold, folds split BY QUESTION -- the honest test for
                           "will it grade questions an instructor uploads later?"
  3. Out-of-domain sanity: tiny toy set from unrelated subjects (biology/physics/CS)

The saved model is trained on the TRAIN split only, so the demo files built from
the held-out test split are not leaked into it.

Usage:
    python src/train_feature_model.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_scorer import FeatureAnswerScorer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    return {
        "pearson_r": round(float(pearsonr(y, p)[0]), 3),
        "qwk": round(float(cohen_kappa_score(y, p, weights="quadratic")), 3),
        "accuracy": round(float((y == p).mean()), 3),
        "n": int(len(y)),
    }


def main():
    train = pd.read_csv(DATA / "real_answers_train.csv")
    test = pd.read_csv(DATA / "real_answers_test.csv")
    full = pd.read_csv(DATA / "real_answers.csv")

    scorer = FeatureAnswerScorer().fit(train)
    results = {"in_domain_test": metrics(test["human_score"], scorer.predict_df(test))}

    # Unseen-question cross-validation (whole dataset, folds grouped by question)
    pred = np.zeros(len(full), dtype=int)
    for tr, te in GroupKFold(5).split(full, full["human_score"], full["question_id"]):
        pred[te] = FeatureAnswerScorer().fit(full.iloc[tr]).predict_df(full.iloc[te])
    results["unseen_question_cv"] = metrics(full["human_score"], pred)
    results["majority_class_accuracy"] = round(float((full["human_score"] == 2).mean()), 3)

    # Out-of-domain sanity check (0-3 toy marks -> incorrect / partial / correct)
    toy = pd.read_csv(DATA / "sample_answers.csv")
    frac = toy["human_score"] / toy["max_score"]
    y_toy = np.where(frac == 0, 0, np.where(frac == 1, 2, 1))
    results["out_of_domain_toy"] = metrics(y_toy, FeatureAnswerScorer().fit(full).predict_df(toy))

    scorer.save()
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / "feature_model_results.json").write_text(json.dumps(results, indent=2))

    print("Domain-agnostic feature model (random forest)\n")
    for k, v in results.items():
        print(f"{k:<26}{v}")
    print("\nSaved -> models/feature_scorer.joblib and eval/feature_model_results.json")


if __name__ == "__main__":
    main()
