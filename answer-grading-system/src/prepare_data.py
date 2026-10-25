"""
Prepares the real short-answer grading dataset for the pipeline.

Source: DigiKlausur/ASAG-Dataset (MPL-2.0 license)
https://github.com/DigiKlausur/ASAG-Dataset

646 exam answers from a graduate Neural Networks course (38 students,
17 questions), each hand-graded 0 (incorrect) / 1 (partial) / 2 (correct)
by a human examiner, with a reference answer per question -- which is
exactly the (reference_answer, student_answer) -> score structure this
pipeline is built around, so no schema changes were needed.

Attribution: cite the ASAG-Dataset repo (and Sultan et al., 2016, whose
alignment/similarity features informed how the dataset was built) in
your report's dataset section.

Usage:
    python src/prepare_data.py
"""

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "raw_asag_dataset.csv"
RAW_URL = "https://raw.githubusercontent.com/DigiKlausur/ASAG-Dataset/master/asag_dataset.csv"

SCHEMA_COLUMNS = ["question_id", "question", "reference_answer", "student_answer", "human_score", "max_score"]


def ensure_raw_data():
    """Download the source dataset if it isn't already present (~12MB)."""
    if RAW_PATH.exists():
        return
    print(f"Downloading source dataset from {RAW_URL} ...")
    DATA_DIR.mkdir(exist_ok=True)
    urllib.request.urlretrieve(RAW_URL, RAW_PATH)
    print("Done.")


def load_and_clean() -> pd.DataFrame:
    ensure_raw_data()
    raw = pd.read_csv(RAW_PATH)

    df = raw[["question_id", "question", "ref_answer", "student_answer", "grades_round"]].copy()
    df.columns = ["question_id", "question", "reference_answer", "student_answer", "human_score"]

    before = len(df)
    df = df.dropna(subset=["student_answer", "reference_answer", "human_score"])
    df["student_answer"] = df["student_answer"].str.strip()
    df = df[df["student_answer"].str.len() > 0]
    dropped = before - len(df)

    df["human_score"] = df["human_score"].astype(int)
    df["max_score"] = 2  # grading scale in this dataset is 0-2

    print(f"Loaded {before} raw rows, dropped {dropped} empty/null rows, kept {len(df)}.")
    return df[SCHEMA_COLUMNS]


def main():
    df = load_and_clean()

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["human_score"]
    )

    df.to_csv(DATA_DIR / "real_answers.csv", index=False)
    train_df.to_csv(DATA_DIR / "real_answers_train.csv", index=False)
    test_df.to_csv(DATA_DIR / "real_answers_test.csv", index=False)

    print(f"\nFull set : {len(df)} rows  -> data/real_answers.csv")
    print(f"Train set: {len(train_df)} rows -> data/real_answers_train.csv")
    print(f"Test set : {len(test_df)} rows -> data/real_answers_test.csv")
    print("\nScore distribution (full set):")
    print(df["human_score"].value_counts().sort_index())


if __name__ == "__main__":
    main()
