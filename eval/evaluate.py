"""
Evaluation harness.

Runs a scorer over the full dataset and reports:
  - Pearson correlation  (linear agreement with human scores)
  - Quadratic Weighted Kappa (the standard metric for ordinal grading
    tasks -- this is what ASAP-SAS itself is judged on, so citing it
    in your report signals you know the field)

Usage:
    python eval/evaluate.py
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from baseline_scorer import BaselineAnswerScorer  # noqa: E402


def evaluate(df: pd.DataFrame, scorer) -> dict:
    predictions = []
    for _, row in df.iterrows():
        result = scorer.score(row["reference_answer"], row["student_answer"], row["max_score"])
        predictions.append(result["predicted_score"])

    df = df.copy()
    df["predicted_score"] = predictions

    pearson_r, p_value = pearsonr(df["human_score"], df["predicted_score"])
    qwk = cohen_kappa_score(df["human_score"], df["predicted_score"], weights="quadratic")
    exact_match = (df["human_score"] == df["predicted_score"]).mean()

    return {
        "pearson_r": round(pearson_r, 3),
        "p_value": round(p_value, 4),
        "quadratic_weighted_kappa": round(qwk, 3),
        "exact_match_accuracy": round(exact_match, 3),
        "n": len(df),
        "predictions": df[["question_id", "human_score", "predicted_score"]],
    }


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    train_path = data_dir / "real_answers_train.csv"
    test_path = data_dir / "real_answers_test.csv"

    if not train_path.exists():
        print("Real dataset not found -- run `python src/prepare_data.py` first.")
        print("Falling back to the tiny sample_answers.csv for a quick sanity check.\n")
        train_path = test_path = data_dir / "sample_answers.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Fit the vectorizer on TRAIN only, evaluate on held-out TEST --
    # no data leakage, so these numbers are trustworthy in the report.
    scorer = BaselineAnswerScorer()
    scorer.fit(list(train_df["reference_answer"]) + list(train_df["student_answer"]))

    results = evaluate(test_df, scorer)

    print(f"=== Baseline (TF-IDF + cosine similarity) -- fit on {len(train_df)} train rows, evaluated on {len(test_df)} held-out test rows ===")
    print(f"n                        : {results['n']}")
    print(f"Pearson r                : {results['pearson_r']}  (p={results['p_value']})")
    print(f"Quadratic Weighted Kappa : {results['quadratic_weighted_kappa']}")
    print(f"Exact match accuracy     : {results['exact_match_accuracy']}")
    print()
    print("Sample predictions (first 10 of the test set):")
    print(results["predictions"].head(10).to_string(index=False))
