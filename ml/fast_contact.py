"""Low-latency contact gate trained on 12 ms real sensor snippets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .config import MODELS_DIR


def fast_contact_features(window: np.ndarray, noise: np.ndarray) -> np.ndarray:
    values = np.asarray(window, dtype=np.float64)
    values = values - np.median(values, axis=0, keepdims=True)
    scale = np.maximum(np.asarray(noise, dtype=np.float64), 0.5)
    rms = np.sqrt(np.mean(values**2, axis=0)) / scale
    peak = np.max(np.abs(values), axis=0) / scale
    line = np.mean(np.abs(np.diff(values, axis=0)), axis=0) / scale
    relative = rms / (np.sum(rms) + 1e-12)
    return np.concatenate([np.log1p(rms), np.log1p(peak), np.log1p(line), relative]).astype(np.float32)


@dataclass
class FastContactGate:
    model: object
    threshold: float

    @classmethod
    def load(cls, path: Path = MODELS_DIR / "fast_contact_real.pkl") -> "FastContactGate":
        if not path.exists():
            raise FileNotFoundError("Fast contact model missing. Run: python -m ml.train_fast_contact")
        artifact = joblib.load(path)
        model = artifact["model"]
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        return cls(model=model, threshold=float(artifact["threshold"]))

    def probability(self, window: np.ndarray, noise: np.ndarray) -> float:
        features = fast_contact_features(window, noise).reshape(1, -1)
        probabilities = self.model.predict_proba(features)[0]
        index = list(self.model.classes_).index("touch")
        return float(probabilities[index])

