"""
Train the domain-agnostic scorer and report three evaluations of its fine-grained marks.

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


def band(f):
    f = np.asarray(f)
    return np.where(f < 0.25, 0, np.where(f < 0.75, 1, 2))


def metrics(y_frac, pred_frac) -> dict:
    """y_frac: teacher mark as a fraction of full marks. pred_frac: model's fine-grained score in [0,1].
    Pearson r and MAE use the fine-grained score; kappa / accuracy compare grade bands."""
    y, p = np.asarray(y_frac, float), np.asarray(pred_frac, float)
    return {
        "pearson_r": round(float(pearsonr(y, p)[0]), 3),
        "mae_fraction_of_marks": round(float(np.abs(y - p).mean()), 3),
        "qwk": round(float(cohen_kappa_score(band(y), band(p), weights="quadratic")), 3),
        "accuracy": round(float((band(y) == band(p)).mean()), 3),
        "distinct_marks_out_of_10": int(len(np.unique(np.round(p * 20) / 2))),
        "n": int(len(y)),
    }


def main():
    train = pd.read_csv(DATA / "real_answers_train.csv")
    test = pd.read_csv(DATA / "real_answers_test.csv")
    full = pd.read_csv(DATA / "real_answers.csv")

    scorer = FeatureAnswerScorer().fit(train)
    results = {"in_domain_test": metrics(test["human_score"] / 2, scorer.predict_fraction_df(test))}

    # Unseen-question cross-validation (whole dataset, folds grouped by question)
    pred = np.zeros(len(full))
    for tr, te in GroupKFold(5).split(full, full["human_score"], full["question_id"]):
        pred[te] = FeatureAnswerScorer().fit(full.iloc[tr]).predict_fraction_df(full.iloc[te])
    results["unseen_question_cv"] = metrics(full["human_score"] / 2, pred)
    results["majority_class_accuracy"] = round(float((full["human_score"] == 2).mean()), 3)

    # Out-of-domain sanity check (toy set graded 0-3)
    toy = pd.read_csv(DATA / "sample_answers.csv")
    y_toy = toy["human_score"] / toy["max_score"]  # teacher marks on a finer 0-3 scale
    results["out_of_domain_toy"] = metrics(y_toy, FeatureAnswerScorer().fit(full).predict_fraction_df(toy))

    scorer.save()
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / "feature_model_results.json").write_text(json.dumps(results, indent=2))

    print("Domain-agnostic feature model (random forest)\n")
    for k, v in results.items():
        print(f"{k:<26}{v}")
    print("\nSaved -> models/feature_scorer.joblib and eval/feature_model_results.json")


if __name__ == "__main__":
    main()
