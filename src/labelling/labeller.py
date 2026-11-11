"""
labeller.py
-----------
Label classifier for Lumen Reader.
Assigns a topic label to each chunk using TF-IDF + Logistic Regression.
Supports: train, predict, save, load.
"""

import os
import pickle
from typing import List, Tuple
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


MODEL_PATH = Path("models/label_classifier.pkl")

# ── Default label taxonomy ─────────────────────────────────────────
# These are starter labels. Add your own by extending SEED_DATA below.

SEED_DATA: List[Tuple[str, str]] = [
    # (text_sample, label)
    ("financial performance revenue profit loss earnings EBITDA",        "Finance"),
    ("risk management compliance regulatory framework audit control",    "Risk"),
    ("governance board directors committee oversight policy",            "Governance"),
    ("AI diligence artificial intelligence machine learning model",      "AI_Diligence"),
    ("legal contract agreement terms liability indemnity clause",        "Legal"),
    ("operations process workflow system infrastructure technology",     "Operations"),
    ("market competition industry analysis sector growth opportunity",   "Market"),
    ("human resources employees talent hiring culture diversity",        "HR"),
    ("environmental sustainability ESG carbon emissions green",          "ESG"),
    ("summary overview introduction executive background context",       "General"),
    ("recommendation action plan next steps conclusion findings",        "Summary"),
    ("data privacy security breach GDPR encryption access control",      "Security"),
]


class LabelClassifier:
    """
    TF-IDF + Logistic Regression label classifier.
    Trained on seed data by default; can be retrained on custom data.
    """

    def __init__(self):
        self.pipeline: Pipeline | None = None
        self.encoder = LabelEncoder()
        self._trained = False

    # ── Public API ─────────────────────────────────────────────────

    def train(self, texts: List[str] = None, labels: List[str] = None):
        """
        Train the classifier.
        If no data passed, trains on built-in SEED_DATA.

        Args:
            texts  : List of text samples (optional)
            labels : Corresponding label strings (optional)
        """
        if texts is None or labels is None:
            texts  = [t for t, _ in SEED_DATA]
            labels = [l for _, l in SEED_DATA]

        # Expand seed data by duplicating with minor variation
        # (ensures logistic regression has enough samples per class)
        texts, labels = self._expand_seed(texts, labels, repeat=6)

        encoded_labels = self.encoder.fit_transform(labels)

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=8000,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=5.0,
                solver="lbfgs",
            )),
        ])

        self.pipeline.fit(texts, encoded_labels)
        self._trained = True
        print(f"[Labeller] Trained on {len(texts)} samples, "
              f"{len(self.encoder.classes_)} labels: "
              f"{list(self.encoder.classes_)}")

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict label for a single text chunk.

        Returns:
            (label_string, confidence_score)
        """
        self._ensure_trained()
        probs   = self.pipeline.predict_proba([text])[0]
        top_idx = probs.argmax()
        label   = self.encoder.inverse_transform([top_idx])[0]
        conf    = round(float(probs[top_idx]), 4)
        return label, conf

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict labels for a list of chunks."""
        self._ensure_trained()
        return [self.predict(t) for t in texts]

    def save(self, path: Path = MODEL_PATH):
        """Persist trained model to disk."""
        self._ensure_trained()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"pipeline": self.pipeline, "encoder": self.encoder}, f)
        print(f"[Labeller] Model saved -> {path}")

    def load(self, path: Path = MODEL_PATH):
        """Load a previously saved model from disk."""
        if not path.exists():
            raise FileNotFoundError(
                f"No saved model at {path}. Call .train() first."
            )
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.pipeline = data["pipeline"]
        self.encoder  = data["encoder"]
        self._trained = True
        print(f"[Labeller] Model loaded <- {path}")

    def label_chunks(self, chunks: List[dict]) -> List[dict]:
        """
        Add 'label' and 'label_confidence' keys to each chunk dict.
        Works directly with chunker.py output.

        Args:
            chunks: List of chunk dicts from chunk_document()

        Returns:
            Same list with 'label' and 'label_confidence' added.
        """
        self._ensure_trained()
        for chunk in chunks:
            label, conf = self.predict(chunk["text"])
            chunk["label"]            = label
            chunk["label_confidence"] = conf
        return chunks

    # ── Private helpers ────────────────────────────────────────────

    def _ensure_trained(self):
        if not self._trained:
            raise RuntimeError(
                "Classifier not trained. Call .train() or .load() first."
            )

    @staticmethod
    def _expand_seed(
        texts: List[str],
        labels: List[str],
        repeat: int = 6,
    ) -> Tuple[List[str], List[str]]:
        """
        Duplicate seed samples to give logistic regression
        enough data per class. Adds minor word shuffling per copy.
        """
        import random
        expanded_t, expanded_l = [], []
        for text, label in zip(texts, labels):
            words = text.split()
            for i in range(repeat):
                shuffled = words.copy()
                random.shuffle(shuffled)
                expanded_t.append(" ".join(shuffled))
                expanded_l.append(label)
        return expanded_t, expanded_l


# ── Singleton for easy import ──────────────────────────────────────

_classifier = LabelClassifier()


def get_classifier(auto_train: bool = True) -> LabelClassifier:
    """
    Return the shared classifier instance.
    Loads from disk if model exists, otherwise trains on seed data.
    """
    if MODEL_PATH.exists():
        _classifier.load()
    elif auto_train:
        _classifier.train()
        _classifier.save()
    return _classifier


# ── Quick test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    clf = LabelClassifier()
    clf.train()
    clf.save()

    test_texts = [
        "The company reported strong revenue growth of 24% year over year with EBITDA margins improving.",
        "Board of directors approved the new governance framework and audit committee charter.",
        "The AI model shows signs of bias in its training data and requires further diligence.",
        "Employee turnover increased significantly in Q3 across engineering and product teams.",
        "Data encryption standards must comply with GDPR and SOC 2 Type II requirements.",
    ]

    print("\n--- Label predictions ---")
    for text in test_texts:
        label, conf = clf.predict(text)
        print(f"  [{label:15s}] ({conf:.2f})  {text[:60]}...")