"""Reusable prediction API for synthetic or real source-specific models."""

from pathlib import Path

import joblib
import numpy as np

from .config import MODELS_DIR, NUM_SENSORS, SAMPLE_RATE
from .features import extract_features
from .source_base import TouchSource


class TouchClassifier:
    def __init__(self, models_dir: Path = MODELS_DIR, source: str = "simulated") -> None:
        suffix = "_real" if source == "real" else ""
        touch_path = models_dir / f"touch_type{suffix}.pkl"
        location_path = models_dir / f"location{suffix}.pkl"
        if not touch_path.exists() or not location_path.exists():
            raise FileNotFoundError(
                f"{source.title()} models not found. Run: python -m ml.train --source {source}"
            )
        self.touch_model, touch_metadata = self._unpack(joblib.load(touch_path))
        self.location_model, location_metadata = self._unpack(joblib.load(location_path))
        presence_path = models_dir / f"touch_presence{suffix}.pkl"
        self.presence_model = None
        self.presence_threshold = 0.5
        if presence_path.exists():
            self.presence_model, presence_metadata = self._unpack(joblib.load(presence_path))
            self.presence_threshold = float(presence_metadata.get("decision_threshold", 0.5))
        for model in (self.touch_model, self.location_model, self.presence_model):
            self._limit_live_workers(model)
        for key in ("sensor_count", "sample_rate", "num_samples", "feature_schema_version"):
            if touch_metadata.get(key) != location_metadata.get(key):
                raise ValueError(f"Touch and location model metadata disagree on {key}.")
        self.metadata = touch_metadata
        self.touch_confidence_floor = float(touch_metadata.get("confidence_floor", 0.0))
        self.location_confidence_floor = float(location_metadata.get("confidence_floor", 0.0))

    @staticmethod
    def _unpack(artifact: object) -> tuple[object, dict[str, object]]:
        if isinstance(artifact, dict) and "model" in artifact:
            return artifact["model"], artifact["metadata"]
        return artifact, {"sensor_count": NUM_SENSORS, "sample_rate": SAMPLE_RATE}

    @staticmethod
    def _limit_live_workers(model: object | None) -> None:
        """Prevent training-time all-core settings from starving the live UI."""
        if model is None:
            return
        candidates = [model, *list(getattr(model, "estimators_", []))]
        for estimator in candidates:
            if hasattr(estimator, "n_jobs"):
                estimator.n_jobs = 1

    @staticmethod
    def _prediction(model: object, features: np.ndarray, confidence_floor: float) -> tuple[str, float]:
        probabilities = model.predict_proba(features)[0]
        best = int(np.argmax(probabilities))
        confidence = float(probabilities[best])
        label = str(model.classes_[best]) if confidence >= confidence_floor else "unknown"
        return label, confidence

    def _features(self, waveform: np.ndarray) -> np.ndarray:
        TouchSource.validate_waveform(waveform, num_sensors=int(self.metadata["sensor_count"]))
        expected_samples = self.metadata.get("num_samples")
        if expected_samples is not None and waveform.shape[0] != int(expected_samples):
            raise ValueError(f"Expected {expected_samples} samples, received {waveform.shape[0]}.")
        return extract_features(waveform, int(self.metadata["sample_rate"])).reshape(1, -1)

    def _core_prediction(self, features: np.ndarray) -> tuple[dict[str, str | float], np.ndarray, np.ndarray]:
        touch_probability = self.touch_model.predict_proba(features)[0]
        location_probability = self.location_model.predict_proba(features)[0]
        touch_best = int(np.argmax(touch_probability))
        location_best = int(np.argmax(location_probability))
        touch_confidence = float(touch_probability[touch_best])
        location_confidence = float(location_probability[location_best])
        result: dict[str, str | float] = {
            "touch_type": str(self.touch_model.classes_[touch_best])
            if touch_confidence >= self.touch_confidence_floor else "unknown",
            "touch_confidence": touch_confidence,
            "location": str(self.location_model.classes_[location_best])
            if location_confidence >= self.location_confidence_floor else "unknown",
            "location_confidence": location_confidence,
        }
        return result, touch_probability, location_probability

    def predict(self, waveform: np.ndarray) -> dict[str, str | float]:
        result, _, _ = self._core_prediction(self._features(waveform))
        return result

    def predict_presence(self, waveform: np.ndarray) -> float:
        """Return touch probability without running the slower gesture/location models."""
        if self.presence_model is None:
            raise RuntimeError("This model set does not include a touch-presence gate.")
        features = self._features(waveform)
        probabilities = self.presence_model.predict_proba(features)[0]
        by_class = dict(zip(self.presence_model.classes_, probabilities, strict=True))
        return float(by_class.get("touch", 0.0))

    def predict_touch_type(self, waveform: np.ndarray) -> tuple[str, float]:
        """Classify only the gesture, for low-latency live use."""
        features = self._features(waveform)
        probabilities = self.touch_model.predict_proba(features)[0]
        best = int(np.argmax(probabilities))
        return str(self.touch_model.classes_[best]), float(probabilities[best])

    def predict_detailed(self, waveform: np.ndarray) -> dict[str, object]:
        """Return the normal prediction plus complete class probabilities."""
        features = self._features(waveform)
        core, touch_probability, location_probability = self._core_prediction(features)
        result: dict[str, object] = dict(core)
        result["touch_probabilities"] = {
            str(label): float(value)
            for label, value in zip(self.touch_model.classes_, touch_probability, strict=True)
        }
        result["location_probabilities"] = {
            str(label): float(value)
            for label, value in zip(self.location_model.classes_, location_probability, strict=True)
        }
        if self.presence_model is not None:
            presence_probability = self.presence_model.predict_proba(features)[0]
            by_class = {
                str(label): float(value)
                for label, value in zip(self.presence_model.classes_, presence_probability, strict=True)
            }
            touch_probability = by_class.get("touch", 0.0)
            result["touch_present"] = bool(touch_probability >= self.presence_threshold)
            result["touch_presence_confidence"] = touch_probability
            result["presence_probabilities"] = by_class
        return result
