"""Train continuous surface localization from directional stroke recordings."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import GroupShuffleSplit

from .body_map import BodySurfaceMap
from .collect_motion_paths import LABELS_PATH, PATH_DIR
from .config import HARDWARE_NUM_SENSORS, MODELS_DIR
from .motion_mapping import spatial_vector
from .preprocessing import subtract_baseline
from .spatial_tracker import SpatialFingerprintTracker


def main() -> None:
    if not LABELS_PATH.exists():
        raise FileNotFoundError("Collect paths first: python -m ml.collect_motion_paths --port COM5")
    rows = pd.read_csv(LABELS_PATH)
    if len(rows) < 40:
        raise ValueError("Collect at least 10 examples for each of the four paths before training.")

    tracker = SpatialFingerprintTracker.from_real_dataset(HARDWARE_NUM_SENSORS, BodySurfaceMap.load())
    features: list[np.ndarray] = []
    targets: list[list[float]] = []
    groups: list[str] = []
    for row in rows.itertuples(index=False):
        waveform = np.load(PATH_DIR / row.file, allow_pickle=False)
        centered, _ = subtract_baseline(waveform, idle_samples=25)
        idle = centered[:25].astype(np.float64)
        noise = np.maximum(1.4826 * np.median(np.abs(idle - np.median(idle, axis=0)), axis=0), 0.5)
        envelope = np.sqrt(np.mean(centered.astype(np.float64) ** 2, axis=1))
        cutoff = np.median(envelope) + 2.5 * np.median(np.abs(envelope - np.median(envelope)))
        active = np.flatnonzero(envelope > cutoff)
        start = int(active[0]) if active.size else 25
        end = int(active[-1]) if active.size else len(waveform) - 1
        if end - start < 100:
            start, end = 25, len(waveform) - 25
        timeline_centers = np.arange(start + 16, end - 15, 10, dtype=int)
        timeline_vectors = []
        for history_center in timeline_centers:
            segment = centered[history_center - 16:history_center + 16].astype(np.float64)
            segment -= np.median(segment, axis=0, keepdims=True)
            rms = np.sqrt(np.mean(segment**2, axis=0))
            timeline_vectors.append(spatial_vector(np.maximum(rms - noise, 0), tracker.sensor_scales))
        timeline_vectors = np.stack(timeline_vectors)
        centers = np.linspace(start + 16, end - 16, 12).astype(int)
        for center in centers:
            timeline_index = int(np.searchsorted(timeline_centers, center, side="right"))
            prefix = timeline_vectors[:max(1, timeline_index)]
            if len(prefix) < 4:
                prefix = np.concatenate([np.repeat(prefix[:1], 4 - len(prefix), axis=0), prefix])
            temporal_bins = [np.mean(chunk, axis=0) for chunk in np.array_split(prefix, 4)]
            progress = float(np.clip((center - start) / max(end - start, 1), 0, 1))
            if row.direction == "back_to_front":
                progress = 1.0 - progress
            u = 0.24 + progress * (0.80 - 0.24)
            v = 0.75 if row.side == "left" else 0.25
            features.append(np.concatenate([
                prefix[-1],
                *temporal_bins,
            ]))
            targets.append([u, np.sin(2 * np.pi * v), np.cos(2 * np.pi * v)])
            groups.append(str(row.file))

    x = np.stack(features)
    y = np.asarray(targets)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_index, test_index = next(splitter.split(x, y, groups))
    u_model = ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=3,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )
    side_labels = np.where(y[:, 1] < 0, "left", "right")
    side_model = ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=43,
        n_jobs=-1,
    )
    u_model.fit(x[train_index], y[train_index, 0])
    side_model.fit(x[train_index], side_labels[train_index])
    predicted_u = u_model.predict(x[test_index])
    predicted_side = side_model.predict(x[test_index])
    true_v = (np.arctan2(y[test_index, 1], y[test_index, 2]) / (2 * np.pi)) % 1.0
    true_side = np.where(true_v > 0.5, "left", "right")
    print(f"Held-out median front/rear error: {median_absolute_error(y[test_index, 0], predicted_u):.3f}")
    print(f"Held-out side accuracy: {np.mean(predicted_side == true_side):.3f}")

    u_model.n_jobs = 1
    side_model.n_jobs = 1
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "u_model": u_model,
            "side_model": side_model,
            "sensor_count": HARDWARE_NUM_SENSORS,
            "feature_schema": 2,
            "sensor_scales": tracker.sensor_scales,
        },
        MODELS_DIR / "spatial_motion_real.pkl",
    )
    print("Saved models/spatial_motion_real.pkl")
    print("Restart the dashboard to load the improved spatial model.")


if __name__ == "__main__":
    main()
