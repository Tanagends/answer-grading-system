"""
Quick end-to-end smoke test -- no dashboard needed.

Usage:
    python run_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402
from trained_scorer import TrainedAnswerScorer  # noqa: E402
from feedback import generate_feedback  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"


def main():
    train_path = DATA_DIR / "real_answers_train.csv"
    test_path = DATA_DIR / "real_answers_test.csv"

    if not train_path.exists():
        print("Real dataset not found -- run `python src/prepare_data.py` first.\n")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    scorer = TrainedAnswerScorer()
    scorer.fit(train_df)

    print(f"Trained on {len(train_df)} real graded answers. Showing 8 held-out test examples:\n")

    for _, row in test_df.head(8).iterrows():
        result = scorer.score(row["reference_answer"], row["student_answer"], row["max_score"])
        fb = generate_feedback(row["reference_answer"], row["student_answer"], result["predicted_score"], row["max_score"])

        print(f"[Q{row['question_id']}] human={row['human_score']}  predicted={result['predicted_score']}  confidence={result['confidence']}")
        print(f"  student answer : {row['student_answer'][:100]}")
        print(f"  feedback       : {fb['summary']}")
        if fb["concepts_missing"]:
            print(f"  missing        : {', '.join(fb['concepts_missing'])}")
        print()


if __name__ == "__main__":
    main()
