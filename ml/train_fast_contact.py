"""Train the 12 ms contact/no-contact gate used by live heat rendering."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from .config import LABELS_PATH, MODELS_DIR, RAW_DATA_DIR
from .fast_contact import fast_contact_features


def main() -> None:
    rows = pd.read_csv(LABELS_PATH)
    rows = rows[(rows["source"] == "real") & (rows["num_sensors"] == 5)]
    features: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    for row in rows.itertuples(index=False):
        waveform = np.load(RAW_DATA_DIR / row.file, allow_pickle=False).astype(np.float64)
        idle_count = min(60, max(12, len(waveform) // 5))
        baseline = np.median(waveform[:idle_count], axis=0)
        idle = waveform[:idle_count] - baseline
        noise = np.maximum(1.4826 * np.median(np.abs(idle - np.median(idle, axis=0)), axis=0), 0.5)
        corrected = waveform - baseline
        candidates: list[tuple[float, np.ndarray]] = []
        for start in range(0, len(corrected) - 11, 6):
            segment = corrected[start:start + 12]
            score = float(np.max(np.max(np.abs(segment), axis=0) / noise))
            candidates.append((score, fast_contact_features(segment, noise)))
        if row.touch_type == "no_touch":
            selected = candidates[::2]
            label = "no_touch"
        else:
            selected = sorted(candidates, key=lambda item: item[0], reverse=True)[:8]
            label = "touch"
        for _, vector in selected:
            features.append(vector)
            labels.append(label)
            groups.append(str(row.file))

    x = np.stack(features)
    y = np.asarray(labels)
    group_array = np.asarray(groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_index, test_index = next(splitter.split(x, y, group_array))
    model = make_pipeline(
        RobustScaler(),
        LogisticRegression(
            C=1.5,
            class_weight="balanced",
            max_iter=2_000,
            random_state=44,
        ),
    )
    train_labels = y[train_index]
    model.fit(x[train_index], train_labels)
    touch_index = list(model.classes_).index("touch")
    probabilities = model.predict_proba(x[test_index])[:, touch_index]
    truth = y[test_index] == "touch"
    candidates = np.linspace(0.25, 0.99, 149)
    valid = []
    for threshold in candidates:
        false_rate = float(np.mean(probabilities[~truth] >= threshold))
        recall = float(np.mean(probabilities[truth] >= threshold))
        if false_rate <= 0.05:
            valid.append((recall, -threshold, threshold, false_rate))
    if valid:
        threshold = max(valid)[2]
    else:
        threshold = max(
            candidates,
            key=lambda value: balanced_accuracy_score(truth, probabilities >= value),
        )
    prediction = probabilities >= threshold
    print(f"Held-out balanced accuracy: {balanced_accuracy_score(truth, prediction):.3f}")
    print(f"Held-out touch recall: {np.mean(prediction[truth]):.3f}")
    print(f"Held-out idle false-positive rate: {np.mean(prediction[~truth]):.3f}")
    print(f"Fast-gate threshold: {threshold:.2f}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "window_samples": 12}, MODELS_DIR / "fast_contact_real.pkl")
    print("Saved models/fast_contact_real.pkl")


if __name__ == "__main__":
    main()
