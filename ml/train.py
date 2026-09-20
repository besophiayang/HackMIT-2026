"""Train robust touch and location ensembles from simulated or real recordings."""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from .config import LABELS_PATH, MODELS_DIR, RAW_DATA_DIR, SAMPLE_RATE, TOUCH_CLASS_GROUPS
from .features import extract_features, feature_names


def _report(title: str, truth: np.ndarray, predictions: np.ndarray) -> None:
    classes = sorted(set(truth))
    print(f"\n{title}")
    print(f"Accuracy: {accuracy_score(truth, predictions):.3f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(truth, predictions):.3f}")
    print(classification_report(truth, predictions, labels=classes, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted)")
    print("Classes:", classes)
    print(confusion_matrix(truth, predictions, labels=classes))


def _metadata_value(rows: pd.DataFrame, column: str, fallback: int | None = None) -> int:
    if column not in rows or rows[column].isna().any():
        if fallback is None:
            raise ValueError(f"Selected samples need a {column} value in labels.csv.")
        return fallback
    values = {int(value) for value in rows[column]}
    if len(values) != 1:
        raise ValueError(f"Selected samples contain mixed {column} values: {sorted(values)}")
    return values.pop()


def _build_model(seed: int) -> VotingClassifier:
    """Fuse a nonlinear tree model with a smooth, scale-normalized kernel model."""
    trees = ExtraTreesClassifier(
        n_estimators=600, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )
    kernel = CalibratedClassifierCV(
        make_pipeline(
            RobustScaler(),
            SVC(C=4.0, gamma="scale", class_weight="balanced", random_state=seed),
        ),
        method="sigmoid",
        cv=3,
    )
    return VotingClassifier(
        estimators=[("extra_trees", trees), ("rbf_svm", kernel)],
        voting="soft", weights=[2, 1], n_jobs=-1,
    )


def _augment(window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Model mounting/ADC variation without changing the label."""
    sample_count, sensor_count = window.shape
    global_gain = rng.uniform(0.75, 1.3)
    sensor_gain = rng.lognormal(mean=0.0, sigma=0.12, size=(1, sensor_count))
    shifted = np.roll(window.astype(np.float64), rng.integers(-12, 13), axis=0)
    noise_scale = np.maximum(np.std(shifted, axis=0, keepdims=True) * 0.025, 0.05)
    augmented = shifted * global_gain * sensor_gain + rng.normal(0, noise_scale, size=(sample_count, sensor_count))
    return augmented.astype(np.float32)


def _confidence_floor(model: VotingClassifier, features: np.ndarray, labels: np.ndarray) -> float:
    probabilities = model.predict_proba(features)
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    correct = predictions == labels
    confidences = np.max(probabilities, axis=1)
    return float(np.clip(np.quantile(confidences[correct], 0.10), 0.45, 0.85)) if np.any(correct) else 0.6


def _presence_threshold(model: VotingClassifier, features: np.ndarray, labels: np.ndarray) -> float:
    """Choose a held-out touch threshold that balances misses and false alarms."""
    touch_index = list(model.classes_).index("touch")
    probabilities = model.predict_proba(features)[:, touch_index]
    candidates = np.linspace(0.20, 0.80, 61)
    scores = [
        balanced_accuracy_score(labels, np.where(probabilities >= threshold, "touch", "no_touch"))
        for threshold in candidates
    ]
    best_score = max(scores)
    best = [threshold for threshold, score in zip(candidates, scores, strict=True) if score == best_score]
    return float(min(best, key=lambda value: abs(value - 0.5)))


def train_models(source: str = "simulated", test_size: float = 0.25, sensor_count_filter: int | None = None) -> None:
    if not LABELS_PATH.exists():
        raise FileNotFoundError("No labels.csv found. Collect or generate data first.")
    all_labels = pd.read_csv(LABELS_PATH)
    required = {"file", "touch_type", "location", "source"}
    missing = required - set(all_labels.columns)
    if missing:
        raise ValueError(f"labels.csv is missing columns: {sorted(missing)}")
    labels = all_labels[all_labels["source"] == source].reset_index(drop=True)
    if sensor_count_filter is not None:
        if "num_sensors" not in labels:
            raise ValueError("labels.csv has no num_sensors column.")
        labels = labels[labels["num_sensors"] == sensor_count_filter].reset_index(drop=True)
    if labels.empty:
        detail = f" with {sensor_count_filter} sensors" if sensor_count_filter is not None else ""
        raise ValueError(f"No {source!r} examples{detail} found in labels.csv. Collect them first.")
    labels["model_touch_type"] = labels["touch_type"].map(TOUCH_CLASS_GROUPS).fillna(labels["touch_type"])
    has_background = bool(np.any(labels["touch_type"].to_numpy() == "no_touch"))
    if source == "real" and not has_background:
        raise ValueError(
            "Real training now needs powered-on no-touch recordings. Run: "
            "python -m ml.collect_background --port COM5 --count 100"
        )

    first = np.load(RAW_DATA_DIR / labels.loc[0, "file"], allow_pickle=False)
    sensor_count = _metadata_value(labels, "num_sensors", first.shape[1])
    sample_rate = _metadata_value(labels, "sample_rate", SAMPLE_RATE if source == "simulated" else None)
    waveforms = [np.load(RAW_DATA_DIR / name, allow_pickle=False).astype(np.float32) for name in labels["file"]]
    expected_shape = waveforms[0].shape
    if any(window.shape != expected_shape for window in waveforms):
        raise ValueError("Selected waveforms must all have the same time and sensor dimensions.")

    joint = labels["model_touch_type"].astype(str) + "|" + labels["location"].astype(str)
    # Consecutive captures are highly correlated. Groups of five stay entirely
    # on one side of the split, preventing optimistic recording-burst leakage.
    occurrence = labels.groupby(["model_touch_type", "location"]).cumcount()
    groups = joint + "|burst_" + (occurrence // 5).astype(str)
    n_splits = max(2, min(5, round(1 / max(0.1, min(test_size, 0.5)))))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_indices, test_indices = next(splitter.split(np.zeros(len(labels)), joint, groups))

    print(f"Examples: {len(labels)} ({sensor_count} sensors at {sample_rate} Hz)")
    print(f"Leakage-resistant train/test sizes: {len(train_indices)}/{len(test_indices)}")
    print(f"Recorded gestures: {sorted(labels['touch_type'].unique())}")
    print(f"Grouped touch classes: {sorted(labels['model_touch_type'].unique())}")
    print(f"Location classes: {sorted(labels['location'].unique())}")

    rng = np.random.default_rng(42)
    train_windows, train_rows = [], []
    for index in train_indices:
        train_windows.append(waveforms[index])
        train_rows.append(index)
        for _ in range(2):
            train_windows.append(_augment(waveforms[index], rng))
            train_rows.append(index)
    train_features = np.stack([extract_features(window, sample_rate) for window in train_windows])
    test_features = np.stack([extract_features(waveforms[index], sample_rate) for index in test_indices])
    train_rows_array = np.asarray(train_rows)
    touch_train_all = labels.loc[train_rows_array, "model_touch_type"].to_numpy()
    location_train_all = labels.loc[train_rows_array, "location"].to_numpy()
    touch_test_all = labels.loc[test_indices, "model_touch_type"].to_numpy()
    location_test_all = labels.loc[test_indices, "location"].to_numpy()
    train_positive = touch_train_all != "no_touch"
    test_positive = touch_test_all != "no_touch"
    if not np.any(train_positive) or not np.any(test_positive):
        raise ValueError("The split contains no touch examples.")

    touch_model, location_model = _build_model(42), _build_model(43)
    touch_model.fit(train_features[train_positive], touch_train_all[train_positive])
    location_model.fit(train_features[train_positive], location_train_all[train_positive])
    positive_test_features = test_features[test_positive]
    touch_test = touch_test_all[test_positive]
    location_test = location_test_all[test_positive]
    touch_predictions = touch_model.predict(positive_test_features)
    location_predictions = location_model.predict(positive_test_features)
    _report("Touch type ensemble (touch windows only)", touch_test, touch_predictions)
    _report("Location ensemble (touch windows only)", location_test, location_predictions)

    presence_model: VotingClassifier | None = None
    presence_threshold: float | None = None
    if has_background:
        presence_train = np.where(touch_train_all == "no_touch", "no_touch", "touch")
        presence_test = np.where(touch_test_all == "no_touch", "no_touch", "touch")
        presence_model = _build_model(41)
        presence_model.fit(train_features, presence_train)
        presence_threshold = _presence_threshold(presence_model, test_features, presence_test)
        touch_index = list(presence_model.classes_).index("touch")
        presence_probability = presence_model.predict_proba(test_features)[:, touch_index]
        presence_predictions = np.where(presence_probability >= presence_threshold, "touch", "no_touch")
        _report("Touch-presence gate", presence_test, presence_predictions)

    metadata = {
        "architecture": "ExtraTrees + robust-scaled RBF-SVM soft-voting ensemble",
        "feature_schema_version": 2,
        "sensor_count": sensor_count,
        "sample_rate": sample_rate,
        "num_samples": expected_shape[0],
        "feature_names": feature_names(sensor_count),
        "source": source,
        "touch_class_groups": TOUCH_CLASS_GROUPS,
    }
    suffix = "_real" if source == "real" else ""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    touch_artifact = {"model": touch_model, "metadata": {**metadata, "confidence_floor": _confidence_floor(touch_model, positive_test_features, touch_test)}}
    location_artifact = {"model": location_model, "metadata": {**metadata, "confidence_floor": _confidence_floor(location_model, positive_test_features, location_test)}}
    joblib.dump(touch_artifact, MODELS_DIR / f"touch_type{suffix}.pkl")
    joblib.dump(location_artifact, MODELS_DIR / f"location{suffix}.pkl")
    if presence_model is not None and presence_threshold is not None:
        presence_artifact = {
            "model": presence_model,
            "metadata": {**metadata, "decision_threshold": presence_threshold},
        }
        joblib.dump(presence_artifact, MODELS_DIR / f"touch_presence{suffix}.pkl")
    print(f"\nSaved {source} ensemble models in {MODELS_DIR}")
    print(f"Touch confidence floor: {touch_artifact['metadata']['confidence_floor']:.2f}")
    print(f"Location confidence floor: {location_artifact['metadata']['confidence_floor']:.2f}")
    if presence_threshold is not None:
        print(f"Touch-presence decision threshold: {presence_threshold:.2f}")
    if source == "simulated":
        print("WARNING: Synthetic-data performance does not predict real-robot performance.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["simulated", "real"], default="simulated")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--sensor-count", type=int, help="Train only recordings with this sensor count")
    args = parser.parse_args()
    train_models(args.source, args.test_size, args.sensor_count)


if __name__ == "__main__":
    main()
