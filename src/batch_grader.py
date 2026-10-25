"""
Batch grading pipeline -- the real-world workflow.

Inputs (CSV or Excel):
  1. Questions file      : question_id, question, [max_marks]
  2. Model answers file  : question_id, model_answer          (or put model_answer in file 1)
  3. Student scripts     : one or many files, in any of these shapes
       long      -> student_id, question_id, answer, [human_marks]
       wide      -> one row per student: student_id, Q1, Q2, Q3 ...
       per-student file -> question_id, answer   (student id taken from the file name)

Outputs: per-answer fine-grained marks + verdict + confidence + feedback, a human-review flag,
per-student and per-question summaries, and (if teacher marks are supplied)
agreement metrics.

CLI:
    python src/batch_grader.py --questions q.csv --model-answers m.csv \
        --scripts s1.csv s2.csv --out results.xlsx
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_scorer import FeatureAnswerScorer  # noqa: E402
from feedback import generate_feedback  # noqa: E402
from preprocessing import tokenize  # noqa: E402

VERDICTS = {0: "Incorrect", 1: "Partially correct", 2: "Correct"}


def band(frac):
    """Fraction of marks -> grade band 0/1/2 (used for the verdict label, feedback and agreement stats)."""
    return np.where(np.asarray(frac) < 0.25, 0, np.where(np.asarray(frac) < 0.75, 1, 2))

ALIASES = {
    "qid": ["question_id", "qid", "q_id", "question_no", "question_number", "q_no", "qno", "q", "id", "no", "number"],
    "question": ["question", "question_text", "prompt", "text"],
    "max_marks": ["max_marks", "max_score", "marks", "total_marks", "max", "out_of", "points"],
    "model_answer": ["model_answer", "reference_answer", "ideal_answer", "answer_key", "key", "model", "reference", "answer"],
    "student": ["student_id", "student", "student_name", "name", "roll_no", "roll", "rollno", "id"],
    "answer": ["student_answer", "answer", "response", "student_response", "script", "text"],
    "human": ["human_marks", "teacher_marks", "actual_marks", "manual_marks", "human_score"],
}


# ----------------------------------------------------------------- reading ---
def read_table(src, name: str | None = None) -> pd.DataFrame:
    """Read CSV / Excel from a path or an uploaded file-like object (safe to call repeatedly)."""
    name = name or getattr(src, "name", str(src))
    if hasattr(src, "getvalue"):
        raw = src.getvalue()
    elif hasattr(src, "read"):
        if hasattr(src, "seek"):
            src.seek(0)
        raw = src.read()
    else:
        raw = Path(src).read_bytes()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if Path(name).suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(io.BytesIO(raw), dtype=object).fillna("")
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.StringIO(raw.decode(enc)), dtype=object, keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {name}")


def _norm_col(c) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def _pick(df: pd.DataFrame, key: str, exclude: set | None = None) -> str | None:
    cols = {_norm_col(c): c for c in df.columns}
    for alias in ALIASES[key]:
        if alias in cols and cols[alias] not in (exclude or set()):
            return cols[alias]
    return None


def norm_qid(x) -> str:
    """'Q1', 'q1', 1, '1.0', 'Question 1' -> '1'."""
    s = str(x).strip().lower()
    s = re.sub(r"\.0+$", "", s)
    s = re.sub(r"^(question|qn|q)[\s._\-]*", "", s)
    return re.sub(r"\s+", "", s)


def _clean(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


# ---------------------------------------------------------------- questions ---
def build_question_table(questions_df: pd.DataFrame, answers_df: pd.DataFrame | None = None,
                         default_max_marks: float = 5, warnings: list | None = None) -> pd.DataFrame:
    warnings = warnings if warnings is not None else []
    q = questions_df.copy()

    qid_c = _pick(q, "qid")
    q_c = _pick(q, "question", exclude={qid_c} if qid_c else None)
    ma_c = _pick(q, "model_answer", exclude={c for c in (qid_c, q_c) if c})
    mm_c = _pick(q, "max_marks")

    if qid_c is None:
        q["__qid"] = [str(i + 1) for i in range(len(q))]
        qid_c = "__qid"
        warnings.append("Questions file has no question_id column -- numbered 1..n by row order.")
    out = pd.DataFrame({
        "question_id": q[qid_c].map(_clean),
        "question": q[q_c].map(_clean) if q_c else "",
        "max_marks": pd.to_numeric(q[mm_c], errors="coerce") if mm_c else np.nan,
        "model_answer": q[ma_c].map(_clean) if ma_c else "",
    })

    if answers_df is not None:
        a = answers_df.copy()
        a_qid = _pick(a, "qid")
        a_ans = _pick(a, "model_answer", exclude={a_qid} if a_qid else None)
        if a_ans is None:
            raise ValueError("Model answers file needs a model_answer column.")
        if a_qid is None:
            if len(a) != len(out):
                raise ValueError("Model answers file has no question_id and a different number of rows than the questions file.")
            warnings.append("Model answers file has no question_id column -- matched to questions by row order.")
            out["model_answer"] = a[a_ans].map(_clean).to_numpy()
        else:
            lookup = {norm_qid(k): _clean(v) for k, v in zip(a[a_qid], a[a_ans])}
            out["model_answer"] = [lookup.get(norm_qid(k), "") for k in out["question_id"]]

    out["qkey"] = out["question_id"].map(norm_qid)
    if out["qkey"].duplicated().any():
        warnings.append("Duplicate question IDs found -- keeping the first of each.")
        out = out.drop_duplicates("qkey")
    missing = out[out["model_answer"] == ""]
    if len(missing):
        warnings.append(f"No model answer for question(s): {', '.join(missing['question_id'])} -- these will be skipped.")
        out = out[out["model_answer"] != ""]
    out["max_marks"] = out["max_marks"].fillna(default_max_marks).astype(float)
    if out.empty:
        raise ValueError("No questions with model answers found.")
    return out.reset_index(drop=True)


# ----------------------------------------------------------------- scripts ---
def parse_scripts(files: list[tuple[str, pd.DataFrame]], questions: pd.DataFrame, warnings: list) -> pd.DataFrame:
    """Normalise every uploaded script file into long format: student_id, qkey, student_answer[, human_marks]."""
    qkeys = set(questions["qkey"])
    frames = []
    for fname, df in files:
        stem = Path(fname).stem
        df = df.copy()
        if df.empty:
            warnings.append(f"{fname}: file is empty -- skipped.")
            continue
        qid_c, ans_c = _pick(df, "qid"), _pick(df, "answer", exclude=None)
        stu_c = _pick(df, "student", exclude={qid_c} if qid_c else None)
        hum_c = _pick(df, "human")
        if qid_c is not None and ans_c is not None and qid_c != ans_c:        # long / per-student
            part = pd.DataFrame({
                "student_id": df[stu_c].map(_clean) if stu_c else stem,
                "qkey": df[qid_c].map(norm_qid),
                "raw_q": df[qid_c].map(_clean),
                "student_answer": df[ans_c].map(_clean),
                "human_marks": pd.to_numeric(df[hum_c], errors="coerce") if hum_c else np.nan,
            })
            if not stu_c:
                warnings.append(f"{fname}: no student column - using the file name '{stem}' as the student ID.")
        else:                                                                  # wide
            stu_c = _pick(df, "student") or df.columns[0]
            qcols = {c: norm_qid(c) for c in df.columns if c != stu_c and norm_qid(c) in qkeys}
            ignored = [c for c in df.columns if c != stu_c and c not in qcols]
            if not qcols:
                warnings.append(f"{fname}: couldn't recognise the layout (need question_id + answer columns, or one column per question ID) -- skipped.")
                continue
            if ignored:
                warnings.append(f"{fname}: ignored columns that don't match a question ID: {', '.join(map(str, ignored))}.")
            long = df.melt(id_vars=[stu_c], value_vars=list(qcols), var_name="col", value_name="student_answer")
            part = pd.DataFrame({
                "student_id": long[stu_c].map(_clean),
                "qkey": long["col"].map(qcols),
                "raw_q": long["col"].astype(str),
                "student_answer": long["student_answer"].map(_clean),
                "human_marks": np.nan,
            })
        frames.append(part)
    if not frames:
        raise ValueError("No usable student script files.")
    scripts = pd.concat(frames, ignore_index=True)
    scripts = scripts[scripts["student_id"] != ""]

    unknown = sorted(set(scripts["qkey"]) - qkeys)
    if unknown:
        raw = scripts.drop_duplicates("qkey").set_index("qkey")["raw_q"]
        warnings.append("Answers for question IDs not in the questions file were skipped: " + ", ".join(raw[u] for u in unknown) + ".")
        scripts = scripts[scripts["qkey"].isin(qkeys)]
    scripts = scripts.drop(columns=["raw_q"])
    dup = scripts.duplicated(["student_id", "qkey"], keep="last")
    if dup.any():
        warnings.append(f"{int(dup.sum())} duplicate student/question answer(s) -- kept the last one.")
        scripts = scripts[~dup]
    return scripts.reset_index(drop=True)


def complete_grid(scripts: pd.DataFrame, questions: pd.DataFrame, warnings: list) -> pd.DataFrame:
    """Every student gets a row for every question; a missing answer counts as blank (0 marks),
    so totals and percentages are always out of the FULL exam."""
    students = scripts["student_id"].drop_duplicates()
    grid = pd.MultiIndex.from_product([students, questions["qkey"]], names=["student_id", "qkey"]).to_frame(index=False)
    full = grid.merge(scripts, on=["student_id", "qkey"], how="left")
    missing = full["student_answer"].isna()
    if missing.any():
        per = full[missing].groupby("student_id").size()
        warnings.append("Missing answers counted as blank (0 marks): " + ", ".join(f"{k} ({v})" for k, v in per.items()) + ".")
    full["student_answer"] = full["student_answer"].fillna("")
    return full


# ------------------------------------------------------------------ grading ---
def grade(questions: pd.DataFrame, scripts: pd.DataFrame, scorer: FeatureAnswerScorer,
          review_threshold: float = 0.55, min_words: int = 3, mark_step: float = 0.5) -> pd.DataFrame:
    """Marks are the model's expected score (probability-weighted over the three grade bands),
    rounded to `mark_step` -- so an out-of-10 answer can get 6, 7.5 or 2, not just 0 / 5 / 10."""
    df = scripts.merge(questions, on="qkey", how="left")
    df["reference_answer"] = df["model_answer"]
    df["blank"] = df["student_answer"].map(lambda s: len(tokenize(s)) == 0)

    proba = scorer.predict_proba_df(df)
    frac = scorer.expected_fraction(proba)
    conf = proba.max(axis=1)
    frac = np.where(df["blank"], 0.0, frac)
    conf = np.where(df["blank"], 1.0, conf)
    marks = np.clip(np.round(frac * df["max_marks"] / mark_step) * mark_step, 0, df["max_marks"])
    cls = band(marks / df["max_marks"])

    short = df["student_answer"].map(lambda s: 0 < len(tokenize(s)) < min_words)
    df["verdict"] = [VERDICTS[int(c)] for c in cls]
    df["awarded_marks"] = np.round(marks, 2)
    df["percentage"] = np.round(marks / df["max_marks"] * 100, 1)
    df["confidence"] = np.round(conf, 3)
    df["needs_review"] = (conf < review_threshold) | short.to_numpy()

    fb = []
    for ref, stu, c, mx, blank in zip(df["reference_answer"], df["student_answer"], cls, df["max_marks"], df["blank"]):
        if blank:
            fb.append("No answer provided.")
            continue
        f = generate_feedback(ref, stu, int(c), 2)
        text = f["summary"]
        if f["concepts_missing"]:
            text += " Missing: " + ", ".join(f["concepts_missing"]) + "."
        fb.append(text)
    df["feedback"] = fb

    cols = ["student_id", "question_id", "question", "max_marks", "student_answer", "awarded_marks",
            "percentage", "verdict", "confidence", "needs_review", "feedback", "model_answer", "human_marks"]
    out = df[cols].copy()
    out["_s"], out["_q"] = out["student_id"].map(_nat_key), out["question_id"].map(_nat_key)
    return out.sort_values(["_s", "_q"]).drop(columns=["_s", "_q"]).reset_index(drop=True)


def _nat_key(v) -> str:
    """Natural-sort key: 'Q2' < 'Q10'."""
    return re.sub(r"\d+", lambda m: m.group().zfill(6), str(v).lower())


# ---------------------------------------------------------------- analytics ---
def student_summary(res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby("student_id", sort=False)
    out = pd.DataFrame({
        "total_marks": g["awarded_marks"].sum().round(2),
        "max_total": g["max_marks"].sum(),
        "answered": g["student_answer"].apply(lambda s: int((s.str.strip() != "").sum())),
        "needs_review": g["needs_review"].sum().astype(int),
    })
    out["percentage"] = (out["total_marks"] / out["max_total"] * 100).round(1)
    out["rank"] = out["percentage"].rank(ascending=False, method="min").astype(int)
    return out.reset_index().sort_values("rank")


def question_summary(res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby("question_id", sort=False)
    out = pd.DataFrame({
        "question": g["question"].first(),
        "max_marks": g["max_marks"].first(),
        "avg_marks": g["awarded_marks"].mean().round(2),
        "avg_percentage": g["percentage"].mean().round(1),
        "correct": g["verdict"].apply(lambda s: int((s == "Correct").sum())),
        "partial": g["verdict"].apply(lambda s: int((s == "Partially correct").sum())),
        "incorrect": g["verdict"].apply(lambda s: int((s == "Incorrect").sum())),
        "needs_review": g["needs_review"].sum().astype(int),
    })
    return out.reset_index()


def evaluate_against_human(res: pd.DataFrame) -> dict | None:
    """Agreement with teacher marks, if the scripts file carried a human_marks column."""
    h = res[res["human_marks"].notna()].copy()
    if len(h) < 5:
        return None
    from scipy.stats import pearsonr
    from sklearn.metrics import cohen_kappa_score

    h_frac = (h["human_marks"] / h["max_marks"]).clip(0, 1)
    p_frac = h["awarded_marks"] / h["max_marks"]
    hc, pc = band(h_frac.to_numpy()), band(p_frac.to_numpy())
    r = float(pearsonr(h_frac, p_frac)[0]) if h_frac.nunique() > 1 and p_frac.nunique() > 1 else float("nan")
    qwk = float(cohen_kappa_score(hc, pc, weights="quadratic")) if len(set(hc) | set(pc)) > 1 else float("nan")
    flagged = h["needs_review"].to_numpy()
    out = {
        "answers_compared": int(len(h)),
        "pearson_r": round(r, 3),
        "qwk": round(qwk, 3),
        "exact_agreement": round(float((hc == pc).mean()), 3),
        "mean_abs_error_marks": round(float((h["human_marks"] - h["awarded_marks"]).abs().mean()), 3),
    }
    if flagged.any() and (~flagged).any():
        out["agreement_when_confident"] = round(float((hc == pc)[~flagged].mean()), 3)
        out["agreement_when_flagged"] = round(float((hc == pc)[flagged].mean()), 3)
    return out


def run_batch(questions_src, answers_src, script_srcs, scorer=None, default_max_marks=5, review_threshold=0.55,
              mark_step=0.5) -> dict:
    """End-to-end. Each *_src is a path or an uploaded file object (anything read_table accepts)."""
    warnings: list[str] = []
    scorer = scorer or FeatureAnswerScorer.load_or_train()
    q_df = read_table(questions_src)
    a_df = read_table(answers_src) if answers_src is not None else None
    questions = build_question_table(q_df, a_df, default_max_marks, warnings)
    files = [(getattr(s, "name", str(s)), read_table(s)) for s in script_srcs]
    scripts = parse_scripts(files, questions, warnings)
    scripts = complete_grid(scripts, questions, warnings)
    results = grade(questions, scripts, scorer, review_threshold, mark_step=mark_step)
    return {
        "results": results,
        "students": student_summary(results),
        "questions": question_summary(results),
        "evaluation": evaluate_against_human(results),
        "warnings": warnings,
    }


def to_excel_bytes(out: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        out["results"].drop(columns=["human_marks"] if out["results"]["human_marks"].isna().all() else []).to_excel(xw, sheet_name="Results", index=False)
        out["students"].to_excel(xw, sheet_name="Student summary", index=False)
        out["questions"].to_excel(xw, sheet_name="Question summary", index=False)
        if out["evaluation"]:
            pd.DataFrame([out["evaluation"]]).T.reset_index().rename(columns={"index": "metric", 0: "value"}).to_excel(xw, sheet_name="Evaluation", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = min(60, max(len(str(c.value)) if c.value is not None else 0 for c in col) + 2)
                ws.column_dimensions[col[0].column_letter].width = width
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description="Batch-grade student scripts against model answers.")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--model-answers", default=None, help="Optional if the questions file already has model answers")
    ap.add_argument("--scripts", required=True, nargs="+")
    ap.add_argument("--default-max-marks", type=float, default=5)
    ap.add_argument("--mark-step", type=float, default=0.5, help="Round marks to the nearest step (0.25, 0.5, 1)")
    ap.add_argument("--out", default="graded_results.xlsx")
    a = ap.parse_args()

    out = run_batch(a.questions, a.model_answers, a.scripts, default_max_marks=a.default_max_marks, mark_step=a.mark_step)
    for w in out["warnings"]:
        print("WARNING:", w)
    Path(a.out).write_bytes(to_excel_bytes(out))
    print(f"\nGraded {len(out['results'])} answers from {out['students'].shape[0]} students -> {a.out}")
    print(out["students"].head(10).to_string(index=False))
    if out["evaluation"]:
        print("\nAgreement with teacher marks:", out["evaluation"])


if __name__ == "__main__":
    main()
