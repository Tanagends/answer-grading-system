"""
Head-to-head model comparison on the real, held-out test set.

Usage:
    python eval/compare_models.py
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from baseline_scorer import BaselineAnswerScorer  # noqa: E402
from trained_scorer import TrainedAnswerScorer  # noqa: E402
from feature_scorer import FeatureAnswerScorer  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def score_with(scorer, df, score_key="predicted_score"):
    preds = []
    for _, row in df.iterrows():
        result = scorer.score(row["reference_answer"], row["student_answer"], row["max_score"])
        preds.append(result[score_key])
    return preds


def metrics(human, predicted):
    r, p = pearsonr(human, predicted)
    qwk = cohen_kappa_score(human, predicted, weights="quadratic")
    acc = (pd.Series(human).values == pd.Series(predicted).values).mean()
    return {"pearson_r": round(r, 3), "p_value": round(p, 4), "qwk": round(qwk, 3), "accuracy": round(acc, 3)}


def main():
    train_df = pd.read_csv(DATA_DIR / "real_answers_train.csv")
    test_df = pd.read_csv(DATA_DIR / "real_answers_test.csv")

    baseline = BaselineAnswerScorer()
    baseline.fit(list(train_df["reference_answer"]) + list(train_df["student_answer"]))
    baseline_preds = score_with(baseline, test_df)
    baseline_metrics = metrics(test_df["human_score"], baseline_preds)

    trained = TrainedAnswerScorer()
    trained.fit(train_df)
    trained_preds = score_with(trained, test_df)
    trained_metrics = metrics(test_df["human_score"], trained_preds)

    print(f"Train: {len(train_df)} rows | Test (held out): {len(test_df)} rows\n")
    header = f"{'Model':<42}{'Pearson r':>10}{'QWK':>8}{'Accuracy':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'Baseline (TF-IDF + threshold)':<42}{baseline_metrics['pearson_r']:>10}{baseline_metrics['qwk']:>8}{baseline_metrics['accuracy']:>10}")
    print(f"{'Trained (TF-IDF + LSA + LogReg)':<42}{trained_metrics['pearson_r']:>10}{trained_metrics['qwk']:>8}{trained_metrics['accuracy']:>10}")

    feature = FeatureAnswerScorer().fit(train_df)
    feature_metrics = metrics(test_df["human_score"], list(feature.predict_df(test_df)))
    print(f"{'Domain-agnostic (features + RandomForest)':<42}{feature_metrics['pearson_r']:>10}{feature_metrics['qwk']:>8}{feature_metrics['accuracy']:>10}")

    try:
        from finetuned_scorer import FineTunedCrossEncoderScorer

        finetuned = FineTunedCrossEncoderScorer()
        finetuned_preds = score_with(finetuned, test_df)
        finetuned_metrics = metrics(test_df["human_score"], finetuned_preds)
        print(f"{'Fine-tuned transformer (CrossEncoder)':<42}{finetuned_metrics['pearson_r']:>10}{finetuned_metrics['qwk']:>8}{finetuned_metrics['accuracy']:>10}")
    except (FileNotFoundError, ImportError) as e:
        print(f"\n(Fine-tuned model not included: {e})")
        print("Run `python src/finetune_crossencoder.py` on a machine with Hugging Face Hub access to add it.")

    print("\nCopy this table straight into your report's Results section.")


if __name__ == "__main__":
    main()
