"""
Wrapper around the fine-tuned cross-encoder (see finetune_crossencoder.py)
so it matches the same .score() interface as baseline_scorer.py and
trained_scorer.py, and can be swapped into eval/compare_models.py or
dashboard/app.py with no other code changes.

Requires a model already fine-tuned and saved by finetune_crossencoder.py
at models/finetuned-crossencoder/ (needs Hugging Face Hub access to run
that step -- see that file's docstring).
"""

from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "finetuned-crossencoder"


class FineTunedCrossEncoderScorer:
    def __init__(self, model_path: Path = MODEL_PATH):
        from sentence_transformers import CrossEncoder  # deferred import -- only needed if this class is used

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"no fine-tuned model at {model_path} -- run `python src/finetune_crossencoder.py` first"
            )
        self.model = CrossEncoder(str(model_path))

    def fit(self, train_df=None):
        # Already fine-tuned and loaded from disk. Kept only for interface
        # parity with BaselineAnswerScorer / TrainedAnswerScorer.
        return self

    def score(self, reference_answer: str, student_answer: str, max_score: int = 2) -> dict:
        raw = float(self.model.predict([(reference_answer, student_answer)])[0])
        raw = max(0.0, min(1.0, raw))
        predicted = max(0, min(max_score, round(raw * max_score)))
        return {
            "predicted_score": predicted,
            "raw_score": round(raw, 4),
            "max_score": max_score,
        }
