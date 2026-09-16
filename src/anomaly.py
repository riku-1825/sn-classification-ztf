from __future__ import annotations
import numpy as np


def prediction_entropy(proba: np.ndarray) -> np.ndarray:
    """Shannon entropy of the predicted class distribution per object.
    Higher entropy = model is less confident = flag for review.
    """
    eps = 1e-12
    return -np.sum(proba * np.log(proba + eps), axis=1)


def max_prob_confidence(proba: np.ndarray) -> np.ndarray:
    """1 - max softmax probability. Higher = less confident."""
    return 1.0 - np.max(proba, axis=1)


def flag_low_confidence(proba: np.ndarray, threshold: float = 0.5, method: str = "entropy"):
    """Return boolean mask of objects to flag as low-confidence /
    candidate anomalies, using a simple threshold on the chosen score.
    `threshold` should be tuned on a held-out validation set, not test.
    """
    if method == "entropy":
        score = prediction_entropy(proba)
    elif method == "max_prob":
        score = max_prob_confidence(proba)
    else:
        raise ValueError(f"Unknown method: {method}")
    return score > threshold, score


class SimpleFeatureAutoencoder:
    """A minimal sklearn-based autoencoder-style reconstruction-error
    score using PCA as a linear stand-in (fast, dependency-light).
    For a nonlinear version, swap in a small torch autoencoder trained
    on the LightGBM feature table.
    """

    def __init__(self, n_components: int = 5):
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=n_components)

    def fit(self, X: np.ndarray):
        self.pca.fit(X)
        return self

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        X_proj = self.pca.transform(X)
        X_rec = self.pca.inverse_transform(X_proj)
        return np.mean((X - X_rec) ** 2, axis=1)
