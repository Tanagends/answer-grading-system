# NLP-Based Automated Answer Grading and Feedback System

Upload a set of questions, their model answers, and a class's answer scripts.
Get marks, per-answer feedback, a human-review queue, class analytics and an
Excel report. Trained and evaluated on real, human-graded exam answers.

## Run it (WSL / Linux / macOS)

```bash
cd answer-grading-system
python3 -m venv venv && source venv/bin/activate     # first time only
pip install -r requirements.txt                       # first time only (re-run: openpyxl is new)
streamlit run dashboard/app.py
```

In the app, tick **"Use the demo files"** for an instant walkthrough, or upload your own.

## The real-world workflow

| You upload | Columns |
|---|---|
| **Questions** (csv/xlsx) | `question_id, question, max_marks` (max_marks optional) |
| **Model answers** (csv/xlsx) | `question_id, model_answer` (optional if the questions file has `model_answer`) |
| **Student scripts** (one or many csv/xlsx) | any mix of: **long** `student_id, question_id, answer` · **wide** one row per student with columns `Q1, Q2, ...` · **per-student file** `question_id, answer` (file name = student ID) |

Optional: add a `human_marks` column to a long-format scripts file and the app
reports agreement between the model and your own marking.

You get: marks per answer (0 / 50% / 100% of `max_marks` for incorrect / partial /
correct), feedback naming missing concepts, a confidence score, a **review queue**
of answers the model is unsure about, per-student totals and ranks, per-question
analytics (which questions the class struggled with), a per-student report, and an
Excel workbook with everything. Unanswered questions score 0 out of the full exam.
Question IDs can be `Q1`, `1` or `Question 1`; bad input produces readable warnings
(missing model answers, unknown question IDs, duplicate answers, blank answers).

Command-line equivalent:

```bash
python src/batch_grader.py --questions sample_inputs/questions.csv \
  --model-answers sample_inputs/model_answers.csv \
  --scripts sample_inputs/scripts_batchA_long.csv sample_inputs/scripts_batchB_wide.csv sample_inputs/S06.csv \
  --out graded_results.xlsx
```

## How it grades (and why it works on new questions)

The first model (TF-IDF + LSA) memorised the *vocabulary* of one exam, so it
defaulted to "2" on any other subject. The grader now uses **domain-agnostic
features measured relative to the model answer**: IDF-weighted concept coverage,
word and phrase overlap, spelling-tolerant character n-gram similarity, and length
ratios. A random forest on those features picks Incorrect / Partial / Correct. Because
nothing depends on subject vocabulary, the same model grades questions it has never seen.
Features are computed per question across the uploaded batch, so they adapt to the subject.

## Results

| Test | Pearson r | QWK | Accuracy | n |
|---|---|---|---|---|
| Held-out answers, known questions | 0.58 | 0.58 | 0.672 | 122 |
| **Entirely unseen questions** (5-fold, split by question) | 0.429 | 0.429 | 0.61 | 607 |
| Other subjects (16-row sanity set: biology, physics, CS) | 0.719 | 0.676 | 0.625 | 16 |

Always predicting "Correct" scores 0.549 accuracy on this data. Model comparison
(`python eval/compare_models.py`), held-out set of 122:

| Model | Pearson r | QWK | Accuracy |
|---|---|---|---|
| Baseline (TF-IDF + threshold) | 0.145 | 0.071 | 0.303 |
| Trained (TF-IDF + LSA + LogReg), domain-locked | 0.406 | 0.398 | 0.664 |
| **Domain-agnostic (features + RandomForest)** | **0.580** | **0.580** | **0.672** |
| Fine-tuned transformer | *run `finetune_crossencoder.py` on a machine with Hugging Face access* | | |

**Be honest about limits in the report:** on questions the model has never seen, accuracy
(0.61) is only modestly above the always-"Correct" baseline (0.55), though agreement
(QWK 0.43) shows it ranks answers sensibly. Training data is a single exam (607 answers),
the 3-level scale is coarse, and single-answer mode is a bit less accurate than batch mode
(no classmates to compare with). The **review queue** is the safeguard: on the demo
data, agreement with teacher marks is ~74% on confident answers vs ~62% on flagged ones.
It is a grading *assistant*, not a replacement for a marker.

## Layout

```
dashboard/app.py            Streamlit app: batch grading + single answer
src/batch_grader.py         parse uploads -> grade -> flag -> summarise -> evaluate -> Excel
src/features.py             domain-agnostic pair features
src/feature_scorer.py       random-forest scorer (save / load)
src/train_feature_model.py  trains + evaluates (in-domain, unseen-question, other-subject)
src/make_sample_inputs.py   builds sample_inputs/ from the held-out split
src/prepare_data.py         downloads + cleans + splits the ASAG dataset
src/baseline_scorer.py, trained_scorer.py, embeddings_scorer.py,
    finetune_crossencoder.py, finetuned_scorer.py   earlier models / transformer upgrade
sample_inputs/              demo files (3 layouts); IDs S01-S06 are synthetic
tests/test_batch_grader.py  python tests/test_batch_grader.py
eval/                       compare_models.py, evaluate.py, feature_model_results.json
models/feature_scorer.joblib  trained on the 485-row TRAIN split only
```

Demo files come from the held-out test split, so the shipped model has not seen them.
Re-train with `python src/train_feature_model.py` (the app also retrains itself if the saved
model is missing or incompatible with your scikit-learn version).

## Dataset

[DigiKlausur/ASAG-Dataset](https://github.com/DigiKlausur/ASAG-Dataset) (MPL-2.0): 646 exam
answers from a graduate Neural Networks course, graded 0/1/2 by a human examiner, with a
reference answer per question; 607 usable after cleaning. License kept in `data/ASAG_DATASET_LICENSE`.

## Next steps

- **Person A (data):** add answers from other subjects (Cambridge/ZIMSEC style) with human grades; retrain and re-run the unseen-question test. This is the biggest accuracy lever.
- **Person B (modeling):** run `finetune_crossencoder.py`, then add the transformer's similarity as one more feature in `features.py` (needs Hugging Face access).
- **Person C (product):** teacher override of flagged marks, `.docx`/PDF script parsing, Shona-English extension.
