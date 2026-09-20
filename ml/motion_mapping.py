"""Shared features for data-driven continuous body-surface localization."""

from __future__ import annotations

import numpy as np


def spatial_vector(sensor_rms: np.ndarray, sensor_scales: np.ndarray) -> np.ndarray:
    """Convert channel energy to a gain-corrected, amplitude-invariant vector."""
    values = np.maximum(np.asarray(sensor_rms, dtype=np.float64), 0) / sensor_scales
    return (values / (np.linalg.norm(values) + 1e-12)).astype(np.float32)

