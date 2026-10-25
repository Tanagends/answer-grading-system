"""
Upgrade path: Sentence-BERT embeddings instead of TF-IDF.

NOTE: this needs `pip install sentence-transformers torch` and a
network that can reach huggingface.co to download model weights.
Sandboxed dev environments often can't reach that domain -- run this
on your own laptop or in Colab, not where you built the baseline.

Same interface as BaselineAnswerScorer on purpose, so eval/evaluate.py
can compare them head-to-head with no other code changes.
"""

from __future__ import annotations


class EmbeddingAnswerScorer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # deferred import

        self.model = SentenceTransformer(model_name)

    def fit(self, corpus: list[str]) -> "EmbeddingAnswerScorer":
        # No fitting needed -- the model is pretrained. Kept for interface
        # parity with BaselineAnswerScorer so both can be swapped in eval code.
        return self

    def similarity(self, reference_answer: str, student_answer: str) -> float:
        from sentence_transformers import util

        embeddings = self.model.encode([reference_answer, student_answer])
        sim = util.cos_sim(embeddings[0], embeddings[1]).item()
        return float(sim)

    def score(self, reference_answer: str, student_answer: str, max_score: int = 3) -> dict:
        sim = self.similarity(reference_answer, student_answer)
        # Embedding similarity rarely drops near 0 for unrelated sentences,
        # so rescale from an empirical floor (~0.15) rather than 0.
        floor = 0.15
        scaled = max(0.0, (sim - floor) / (1 - floor))
        predicted = max(0, min(max_score, round(scaled * max_score)))
        return {
            "similarity": round(sim, 4),
            "predicted_score": predicted,
            "max_score": max_score,
        }


# --- Fine-tuning (Person B's actual next step) ---
#
# See finetune_crossencoder.py for the real, working fine-tuning script:
# it trains a CrossEncoder end-to-end on labeled (reference, student, score)
# triples using the Hugging Face + sentence-transformers training API.
# That script is the one referenced in your report's methodology section --
# this file (EmbeddingAnswerScorer) is only the frozen-embeddings baseline
# above it.
