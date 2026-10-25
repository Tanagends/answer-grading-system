"""
Fine-tune a transformer cross-encoder to predict answer scores directly
from (reference_answer, student_answer) pairs.

This is the real fine-tuning step your abstract describes -- unlike
trained_scorer.py (a classical ML classifier on top of frozen TF-IDF/LSA
features), this updates the transformer's own weights via backprop on
your labeled data. It's the flagship technical result for your report.

Needs Hugging Face Hub access to download the starting checkpoint, so
run this on a laptop or in Colab -- NOT in a network-locked-down
sandbox. A free Colab GPU runtime finishes this in a couple of minutes
on a dataset this size; CPU also works, just slower.

Usage:
    python src/finetune_crossencoder.py
"""

from pathlib import Path

import pandas as pd
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from torch.utils.data import DataLoader

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_OUT = Path(__file__).resolve().parent.parent / "models" / "finetuned-crossencoder"

# Pretrained for sentence-pair similarity regression on STS-B -- a strong
# starting point for fine-tuning further on a ~500-row dataset rather than
# starting from a plain, un-adapted transformer checkpoint.
BASE_MODEL = "cross-encoder/stsb-distilroberta-base"
EPOCHS = 4
BATCH_SIZE = 16


def to_examples(df: pd.DataFrame) -> list[InputExample]:
    """Each row becomes a labeled (reference, student) pair, label normalized to [0,1]."""
    examples = []
    for _, row in df.iterrows():
        label = row["human_score"] / row["max_score"]
        examples.append(InputExample(texts=[row["reference_answer"], row["student_answer"]], label=float(label)))
    return examples


def main():
    train_df = pd.read_csv(DATA_DIR / "real_answers_train.csv")
    test_df = pd.read_csv(DATA_DIR / "real_answers_test.csv")

    train_examples = to_examples(train_df)
    test_examples = to_examples(test_df)

    print(f"Fine-tuning {BASE_MODEL} on {len(train_examples)} pairs ({EPOCHS} epochs)...")
    model = CrossEncoder(BASE_MODEL, num_labels=1)  # regression head, sigmoid output in [0,1]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    evaluator = CECorrelationEvaluator.from_input_examples(test_examples, name="test")

    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=EPOCHS,
        warmup_steps=int(len(train_dataloader) * EPOCHS * 0.1),
        output_path=str(MODEL_OUT),
        show_progress_bar=True,
    )

    print(f"\nFine-tuned model saved to {MODEL_OUT}")
    print("Run `python eval/compare_models.py` -- it will pick this up automatically as a third row.")


if __name__ == "__main__":
    main()
