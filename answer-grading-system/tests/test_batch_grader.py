"""Run with:  python -m pytest tests -q     (or: python tests/test_batch_grader.py)"""
import io
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from batch_grader import run_batch, to_excel_bytes  # noqa: E402

S = ROOT / "sample_inputs"


def _csv(df, name):
    b = io.BytesIO(df.to_csv(index=False).encode()); b.name = name
    return b


def _xlsx(df, name):
    b = io.BytesIO(); df.to_excel(b, index=False); b.seek(0); b.name = name
    return b


def test_demo_files_all_formats():
    out = run_batch(S / "questions.csv", S / "model_answers.csv",
                    [S / "scripts_batchA_long.csv", S / "scripts_batchB_wide.csv", S / "S06.csv"])
    assert out["students"].shape[0] == 6 and len(out["results"]) == 6 * 16
    assert out["evaluation"] and out["evaluation"]["pearson_r"] > 0.4
    assert (out["results"]["awarded_marks"] <= out["results"]["max_marks"]).all()
    assert len(to_excel_bytes(out)) > 1000


def test_blank_and_missing_answers_score_zero():
    out = run_batch(S / "questions.csv", S / "model_answers.csv", [S / "S06.csv"])
    blank = out["results"][out["results"]["student_answer"] == ""]
    assert len(blank) == 4 and (blank["awarded_marks"] == 0).all()
    assert (blank["feedback"] == "No answer provided.").all()


def test_excel_numeric_ids_and_other_subject():
    """Upload a different subject entirely, with numeric IDs, in Excel -- scores must still vary."""
    q = pd.DataFrame({"No": [1, 2], "Question": ["What is photosynthesis?", "What is a variable?"], "Marks": [4, 6]})
    m = pd.DataFrame({"No": [1, 2], "Answer": [
        "Photosynthesis is the process by which green plants use sunlight, water and carbon dioxide to produce glucose and oxygen.",
        "A variable is a named storage location in memory that holds a value which can change while the program runs."]})
    scripts = pd.DataFrame({"Student": ["Ann", "Ann", "Bob", "Bob", "Cy", "Cy"], "Question No": [1, 2] * 3,
                            "Response": ["Plants use sunlight water and carbon dioxide to make glucose and release oxygen.",
                                         "A variable is a named place in memory that stores a value that can change as the program runs.",
                                         "Plants make food from the sun.", "A variable stores data.",
                                         "It is a reaction in animals.", "Variables are used in loops."]})
    out = run_batch(_xlsx(q, "q.xlsx"), _xlsx(m, "m.xlsx"), [_csv(scripts, "s.csv")])
    r = out["results"]
    assert r["verdict"].nunique() >= 2, "all answers got the same verdict"
    ann = out["students"].set_index("student_id").loc["Ann", "percentage"]
    cy = out["students"].set_index("student_id").loc["Cy", "percentage"]
    assert ann > cy
    assert set(r["max_marks"]) == {4.0, 6.0}


def test_warnings_for_bad_input():
    q = pd.DataFrame({"question_id": ["Q1", "Q2"], "question": ["a", "b"]})
    m = pd.DataFrame({"question_id": ["Q1"], "model_answer": ["The heart pumps blood around the body."]})
    s = pd.DataFrame({"student_id": ["A", "A", "A"], "question_id": ["Q1", "Q1", "Q9"],
                      "answer": ["heart pumps blood", "the heart pumps blood round the body", "x"]})
    out = run_batch(_csv(q, "q.csv"), _csv(m, "m.csv"), [_csv(s, "s.csv")])
    text = " ".join(out["warnings"])
    assert "Q2" in text and "Q9" in text and "duplicate" in text.lower()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
