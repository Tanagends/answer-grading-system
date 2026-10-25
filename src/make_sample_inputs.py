"""
Builds demo upload files in sample_inputs/ from the HELD-OUT test split only
(the shipped model never saw these answers).

Student IDs are synthetic (S01..S06): the source dataset is anonymised, so
answers are assigned to students by order within each question. Teacher marks
are the dataset's human grades (0/1/2) rescaled to marks out of 10, just to
show a realistic marks scale.

Three different script layouts are produced on purpose, so you can upload them
together and see that the grader accepts all of them:
  scripts_batchA_long.csv   long format + teacher marks  (S01-S03)
  scripts_batchB_wide.csv   one row per student, one column per question (S04-S05)
  S06.csv                   a single student's script (id taken from the file name; skipped questions count as 0)

Usage:
    python src/make_sample_inputs.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sample_inputs"
MAX_MARKS = 10


def main():
    OUT.mkdir(exist_ok=True)
    t = pd.read_csv(ROOT / "data" / "real_answers_test.csv")
    t["qid"] = "Q" + t["question_id"].astype(str)
    t["k"] = t.groupby("qid").cumcount()
    t = t[(t["k"] < 6) & (t.groupby("qid")["qid"].transform("size") >= 5)].copy()  # drop questions with <5 held-out answers
    t["student_id"] = "S" + (t["k"] + 1).astype(str).str.zfill(2)
    t["human_marks"] = t["human_score"] / 2 * MAX_MARKS

    qs = t.drop_duplicates("qid").sort_values("question_id")
    qs[["qid", "question"]].rename(columns={"qid": "question_id"}).assign(max_marks=MAX_MARKS).to_csv(OUT / "questions.csv", index=False)
    qs[["qid", "reference_answer"]].rename(columns={"qid": "question_id", "reference_answer": "model_answer"}).to_csv(OUT / "model_answers.csv", index=False)

    a = t[t["student_id"].isin(["S01", "S02", "S03"])]
    a[["student_id", "qid", "student_answer", "human_marks"]].rename(
        columns={"qid": "question_id", "student_answer": "answer"}
    ).to_csv(OUT / "scripts_batchA_long.csv", index=False)

    b = t[t["student_id"].isin(["S04", "S05"])]
    wide = b.pivot(index="student_id", columns="qid", values="student_answer").reset_index()
    order = ["student_id"] + [f"Q{i}" for i in range(1, 18) if f"Q{i}" in wide.columns]
    wide[order].fillna("").to_csv(OUT / "scripts_batchB_wide.csv", index=False)

    c = t[t["student_id"] == "S06"]
    c[["qid", "student_answer"]].rename(columns={"qid": "question_id", "student_answer": "answer"}).to_csv(OUT / "S06.csv", index=False)

    print("Wrote:", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
