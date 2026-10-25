"""
Text preprocessing utilities for the answer grading pipeline.

Deliberately dependency-light (no NLTK corpus downloads) so it runs
anywhere with no internet access beyond pip install. Swap in spaCy /
NLTK's WordNetLemmatizer later if you want richer lemmatization --
that needs `nltk.download(...)`, which requires network access, so do
that on a laptop/Colab, not inside a locked-down sandbox.
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"[a-zA-Z']+")

# A handful of common suffixes for crude stemming (Porter-lite).
_SUFFIXES = ("ing", "edly", "ed", "ly", "es", "s")


def clean_text(text: str) -> str:
    """Lowercase and strip anything that isn't a letter or apostrophe."""
    text = text.lower()
    tokens = _TOKEN_RE.findall(text)
    return " ".join(tokens)


def simple_stem(token: str) -> str:
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    return token


def tokenize(text: str, remove_stopwords: bool = True, stem: bool = True) -> list[str]:
    cleaned = clean_text(text)
    tokens = cleaned.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    if stem:
        tokens = [simple_stem(t) for t in tokens]
    return tokens


def preprocess(text: str) -> str:
    """Full pipeline: clean -> tokenize -> stopword removal -> stem -> rejoin."""
    return " ".join(tokenize(text))


if __name__ == "__main__":
    sample = "Photosynthesis is the process by which plants use sunlight!"
    print("raw:      ", sample)
    print("processed:", preprocess(sample))
