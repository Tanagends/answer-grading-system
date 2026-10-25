"""
Feedback generation.

Approach: treat the top TF-IDF terms of the reference answer as its
"key concepts". Any of those terms absent from the student's answer
get surfaced as a missing-concept list. Simple, explainable, and a
fair starting point -- Person C's job is to make this smarter over
the semester (e.g. weight by rubric-tagged keywords, or use
embedding-level concept clustering instead of raw terms).
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import preprocess


def key_concepts(reference_answer: str, top_n: int = 6) -> list[str]:
    """Pull the top_n highest-weighted terms out of the reference answer."""
    processed = preprocess(reference_answer)
    if not processed.strip():
        return []
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([processed])
    scores = matrix.toarray()[0]
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    return [term for term, score in ranked[:top_n] if score > 0]


def generate_feedback(reference_answer: str, student_answer: str, predicted_score: int, max_score: int) -> dict:
    concepts = key_concepts(reference_answer)
    student_tokens = set(preprocess(student_answer).split())

    missing = [] if predicted_score >= max_score else [c for c in concepts if c not in student_tokens]
    covered = [c for c in concepts if c in student_tokens]

    if predicted_score == max_score:
        summary = "Strong answer -- covers the key concepts from the reference answer."
    elif predicted_score == 0:
        summary = "This answer does not appear to address the question."
    elif missing:
        summary = "Partially correct -- some key concepts are missing or underdeveloped."
    else:
        summary = "Partially correct -- concepts are present but explanation could be more complete."

    return {
        "summary": summary,
        "concepts_covered": covered,
        "concepts_missing": missing,
    }


if __name__ == "__main__":
    ref = "Photosynthesis is the process by which green plants use sunlight water and carbon dioxide to produce glucose and oxygen."
    weak_answer = "Plants make food from the sun."
    fb = generate_feedback(ref, weak_answer, predicted_score=1, max_score=3)
    print(fb)
