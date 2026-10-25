"""
Answer Grading Assistant -- real-world workflow.

    streamlit run dashboard/app.py

Tab 1  Batch grading : upload questions + model answers + many student scripts,
                       get marks, feedback, a human-review queue, analytics, export.
Tab 2  Single answer : quick one-off check.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from batch_grader import run_batch, to_excel_bytes  # noqa: E402
from feature_scorer import FeatureAnswerScorer  # noqa: E402
from feedback import generate_feedback  # noqa: E402

DEMO = ROOT / "sample_inputs"
FILE_TYPES = ["csv", "xlsx"]

EXAMPLES = {
    "Example A (human score: 0)": {
        "question": "Explain the Bias Variance Dilemma!",
        "reference_answer": "Bias-variance dilemma is a principle supervised learning problem. The dilemma arises due to the variance of data and bias of model. When there is high bias, the model fits the training data perfectly but suffers from high variance, when the bias is low the variance reduces but the model doesn\u2019t fit the data well. This dilemma makes the generalizability difficult to achieve.",
        "student_answer": "High value of bias means netowrk is unable to learn the data whereas higher variance means its difficult to learn the training data successfully.",
    },
    "Example B (human score: 1)": {
        "question": "Explain back propagation, use the correct technical terms!",
        "reference_answer": "Backpropagation lowers the error of a MLP level by level recursively backwards. It back propagates an error from the last layer to the first layer by updating the weights. The updates are determined by the local gradient at each level which is computed by partial derivatives of the error and chain rule.",
        "student_answer": "Back propagation is a process to make adjustment to the weights of a neural network in a way that minimizes the average squared error of the training data. It uses steepest decent method. In each step it moves towards the direction that gives maximum decrease of the error.",
    },
    "Example C (human score: 2)": {
        "question": "When learning using steepest descent, explain the role of the learning rate? What is a danger?",
        "reference_answer": "Learning rate controls the speed of the convergence. When the learning rate is low, the convergence is overdamped and slow. When the learning rate is high, the convergence is underdamped and follows a zigzagging path. When the learning rate exceeds a critical value learning becomes unstable.",
        "student_answer": "Learning rate in steepest descent can directly affect the convergence of the algorithm. If the learning rate is very small then algorithm can take long time to converge i.e response is overdamped. But if the learning rate is made very high then we may observe zig-zagging (oscillatory) behaviour and sometimes algorithm may fail to converge (underdamped response).",
    },
    "Write your own": {"question": "", "reference_answer": "", "student_answer": ""},
}


@st.cache_resource(show_spinner="Loading grading model...")
def load_scorer():
    return FeatureAnswerScorer.load_or_train()


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


st.set_page_config(page_title="Answer Grading Assistant", page_icon="📝", layout="wide")
st.title("📝 Answer Grading Assistant")
st.caption("Upload questions, model answers and student scripts. Get marks, feedback, a review queue and class analytics.")
scorer = load_scorer()

with st.sidebar:
    st.header("Settings")
    default_max = st.number_input("Default max marks per question", 1.0, 100.0, 5.0, 1.0,
                                  help="Used only for questions whose file has no max_marks column.")
    threshold = st.slider("Flag for review below confidence", 0.34, 0.90, 0.55, 0.01,
                          help="Answers the model is unsure about are flagged so a teacher can check them.")
    st.divider()
    st.caption("**How it grades:** each answer is compared with the model answer (concept coverage, "
               "wording overlap, spelling-tolerant similarity, length) and a classifier trained on "
               "607 real graded exam answers picks *Incorrect / Partially correct / Correct*. "
               "Marks = 0 / 50% / 100% of the question's max marks.")

tab_batch, tab_single = st.tabs(["📚 Batch grading", "✏️ Single answer"])

# ------------------------------------------------------------------ batch ---
with tab_batch:
    use_demo = False
    if DEMO.exists():
        use_demo = st.toggle("Use the demo files (held-out exam answers, 16 questions, 6 students, 3 file layouts)")

    if use_demo:
        q_src, a_src = DEMO / "questions.csv", DEMO / "model_answers.csv"
        s_srcs = [DEMO / "scripts_batchA_long.csv", DEMO / "scripts_batchB_wide.csv", DEMO / "S06.csv"]
    else:
        c1, c2 = st.columns(2)
        q_src = c1.file_uploader("① Questions file", type=FILE_TYPES,
                                 help="Columns: question_id, question, max_marks (optional)")
        a_src = c2.file_uploader("② Model answers file", type=FILE_TYPES,
                                 help="Columns: question_id, model_answer. Optional if the questions file already has model_answer.")
        s_srcs = st.file_uploader("③ Student scripts (one or many files)", type=FILE_TYPES, accept_multiple_files=True,
                                  help="Long, wide, or one-file-per-student layout - see the formats below.")

    with st.expander("File formats & templates"):
        st.markdown(
            "**Questions** `question_id, question, max_marks`  \n"
            "**Model answers** `question_id, model_answer`  \n"
            "**Student scripts** - any mix of these, uploaded together:  \n"
            "- *Long:* `student_id, question_id, answer` (+ optional `human_marks` to measure agreement with a teacher)  \n"
            "- *Wide:* one row per student: `student_id, Q1, Q2, Q3, ...`  \n"
            "- *Per-student file:* `question_id, answer` - the file name becomes the student ID  \n\n"
            "Question IDs may be written `Q1`, `1`, or `Question 1`. Unanswered questions score 0."
        )
        t1, t2, t3, t4 = st.columns(4)
        t1.download_button("questions.csv", csv_bytes(pd.DataFrame({"question_id": ["Q1", "Q2"], "question": ["...", "..."], "max_marks": [5, 10]})), "questions_template.csv")
        t2.download_button("model_answers.csv", csv_bytes(pd.DataFrame({"question_id": ["Q1", "Q2"], "model_answer": ["...", "..."]})), "model_answers_template.csv")
        t3.download_button("scripts (long).csv", csv_bytes(pd.DataFrame({"student_id": ["S01", "S01"], "question_id": ["Q1", "Q2"], "answer": ["...", "..."], "human_marks": ["", ""]})), "scripts_long_template.csv")
        t4.download_button("scripts (wide).csv", csv_bytes(pd.DataFrame({"student_id": ["S01", "S02"], "Q1": ["...", "..."], "Q2": ["...", "..."]})), "scripts_wide_template.csv")

    ready = q_src is not None and len(s_srcs) > 0
    if st.button("Grade scripts", type="primary", disabled=not ready):
        try:
            with st.spinner("Grading..."):
                st.session_state["out"] = run_batch(q_src, a_src, s_srcs, scorer=scorer,
                                                    default_max_marks=default_max, review_threshold=threshold)
        except Exception as e:  # show a readable message instead of a traceback
            st.session_state.pop("out", None)
            st.error(f"Couldn't grade these files: {e}")

    out = st.session_state.get("out")
    if out:
        res, stu, qs, ev = out["results"], out["students"], out["questions"], out["evaluation"]
        for w in out["warnings"]:
            st.warning(w)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Students", len(stu))
        m2.metric("Answers graded", len(res))
        m3.metric("Class average", f"{stu['percentage'].mean():.1f}%")
        flagged = int(res["needs_review"].sum())
        m4.metric("Flagged for review", f"{flagged} ({flagged / max(len(res), 1):.0%})")

        d1, d2, _ = st.columns([1, 1, 4])
        d1.download_button("⬇ Excel report", to_excel_bytes(out), "graded_results.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        d2.download_button("⬇ CSV", csv_bytes(res), "graded_results.csv")

        v_students, v_questions, v_answers, v_review, v_eval, v_report = st.tabs(
            ["Students", "Questions", "All answers", "Review queue", "Evaluation", "Student report"])

        with v_students:
            st.dataframe(stu, hide_index=True, column_config={
                "percentage": st.column_config.ProgressColumn("percentage", min_value=0, max_value=100, format="%.1f%%")})
            st.bar_chart(stu.set_index("student_id")["percentage"])

        with v_questions:
            st.caption("Questions with a low class average or many flagged answers are worth a second look.")
            st.dataframe(qs, hide_index=True, column_config={
                "avg_percentage": st.column_config.ProgressColumn("avg_percentage", min_value=0, max_value=100, format="%.1f%%")})
            st.bar_chart(qs.set_index("question_id")[["correct", "partial", "incorrect"]],
                         color=["#2e7d32", "#f9a825", "#c62828"])

        with v_answers:
            f1, f2, f3 = st.columns([2, 2, 1])
            pick_s = f1.multiselect("Student", list(stu["student_id"]))
            pick_q = f2.multiselect("Question", list(dict.fromkeys(res["question_id"])))
            only_flag = f3.checkbox("Only flagged")
            view = res
            if pick_s:
                view = view[view["student_id"].isin(pick_s)]
            if pick_q:
                view = view[view["question_id"].isin(pick_q)]
            if only_flag:
                view = view[view["needs_review"]]
            st.dataframe(view.drop(columns=["model_answer", "question", "percentage"]), hide_index=True, column_config={
                "confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1, format="%.2f")})

        with v_review:
            queue = res[res["needs_review"]].sort_values("confidence")
            st.caption(f"{len(queue)} answer(s) the model is unsure about (low confidence or very short). "
                       "Check these by hand - the rest can be trusted more.")
            for _, r in queue.head(25).iterrows():
                with st.expander(f"{r['student_id']} · {r['question_id']} · {r['verdict']} ({r['awarded_marks']:g}/{r['max_marks']:g}) · confidence {r['confidence']:.2f}"):
                    st.markdown(f"**Question:** {r['question']}")
                    st.markdown(f"**Model answer:** {r['model_answer']}")
                    st.markdown(f"**Student answer:** {r['student_answer'] or '*(blank)*'}")
                    st.markdown(f"**Feedback:** {r['feedback']}")
            if len(queue) > 25:
                st.caption(f"Showing the 25 least-confident of {len(queue)}. The Excel report has all of them.")

        with v_eval:
            if ev:
                st.markdown(f"**Agreement with the teacher marks you supplied** ({ev['answers_compared']} answers)")
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Pearson r", ev["pearson_r"])
                e2.metric("Quadratic κ", ev["qwk"])
                e3.metric("Same grade band", f"{ev['exact_agreement']:.0%}")
                e4.metric("Avg. mark error", f"{ev['mean_abs_error_marks']:.2f}")
                if "agreement_when_confident" in ev:
                    st.info(f"Agreement is **{ev['agreement_when_confident']:.0%}** on answers the model was confident about "
                            f"vs **{ev['agreement_when_flagged']:.0%}** on flagged ones - which is why the review queue exists.")
            else:
                st.info("Add a `human_marks` column to a long-format scripts file to measure agreement with a teacher's marking.")
            bench = ROOT / "eval" / "feature_model_results.json"
            if bench.exists():
                b = json.loads(bench.read_text())
                st.markdown("**Model benchmark (from training)**")
                st.dataframe(pd.DataFrame([
                    {"test": "Held-out answers, known questions", **{k: b["in_domain_test"][k] for k in ("pearson_r", "qwk", "accuracy", "n")}},
                    {"test": "Entirely unseen questions (5-fold)", **{k: b["unseen_question_cv"][k] for k in ("pearson_r", "qwk", "accuracy", "n")}},
                    {"test": "Different subjects (small sanity set)", **{k: b["out_of_domain_toy"][k] for k in ("pearson_r", "qwk", "accuracy", "n")}},
                ]), hide_index=True)
                st.caption(f"Always predicting 'Correct' would score {b['majority_class_accuracy']:.0%} accuracy on this data.")

        with v_report:
            who = st.selectbox("Student", list(stu["student_id"]))
            rep = res[res["student_id"] == who][["question_id", "question", "student_answer", "awarded_marks", "max_marks", "verdict", "feedback"]]
            row = stu[stu["student_id"] == who].iloc[0]
            st.markdown(f"### {who}: {row['total_marks']:g} / {row['max_total']:g} ({row['percentage']}%) - rank {row['rank']}")
            st.dataframe(rep, hide_index=True)
            st.download_button("⬇ Download this student's report", csv_bytes(rep), f"report_{who}.csv")

# ----------------------------------------------------------------- single ---
with tab_single:
    st.caption("Grade one answer. Slightly less accurate than batch grading, because there are no classmates' answers "
               "to compare against.")
    choice = st.selectbox("Load an example", list(EXAMPLES.keys()))
    ex = EXAMPLES[choice]
    with st.form("single_form"):
        question = st.text_input("Question", value=ex["question"])
        reference_answer = st.text_area("Model answer", value=ex["reference_answer"], height=110)
        student_answer = st.text_area("Student's answer", value=ex["student_answer"], height=110)
        max_marks = st.number_input("Max marks", min_value=1.0, max_value=100.0, value=10.0)
        go = st.form_submit_button("Grade answer")
    if go:
        if not reference_answer.strip() or not student_answer.strip():
            st.warning("Please fill in both the model answer and the student's answer.")
        else:
            one = pd.DataFrame({"question_id": ["q"], "reference_answer": [reference_answer], "student_answer": [student_answer]})
            proba = scorer.predict_proba_df(one)[0]
            cls = int(proba.argmax())
            fb = generate_feedback(reference_answer, student_answer, cls, 2)
            c1, c2, c3 = st.columns(3)
            c1.metric("Marks", f"{cls / 2 * max_marks:g} / {max_marks:g}")
            c2.metric("Verdict", ["Incorrect", "Partially correct", "Correct"][cls])
            c3.metric("Confidence", f"{proba.max():.2f}")
            st.write(fb["summary"])
            if fb["concepts_covered"]:
                st.success("Concepts covered: " + ", ".join(fb["concepts_covered"]))
            if fb["concepts_missing"]:
                st.error("Concepts missing: " + ", ".join(fb["concepts_missing"]))
            m = re.search(r"human score: (\d)", choice)
            if m:  # built-in examples carry the real teacher grade, so agreement/disagreement is visible
                st.caption(f"Teacher's grade for this example: **{['Incorrect', 'Partially correct', 'Correct'][int(m.group(1))]}**. "
                           "The model is right about two-thirds of the time on held-out answers, not always.")
