"""Reusable preprocessing for both training and live inference."""

import numpy as np

from .source_base import TouchSource


def subtract_baseline(
    waveform: np.ndarray, idle_samples: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract an empirical per-sensor median baseline.

    The beginning of each window is assumed to contain idle readings. Capture
    tools include pre-trigger samples for this purpose. If ``idle_samples`` is
    omitted, the first 20 percent of the window is used.
    """
    TouchSource.validate_waveform(waveform)
    if idle_samples is None:
        idle_samples = max(1, waveform.shape[0] // 5)
    if not 1 <= idle_samples <= waveform.shape[0]:
        raise ValueError("idle_samples must be within the waveform length.")
    baseline = np.median(waveform[:idle_samples], axis=0).astype(np.float32)
    corrected = (waveform - baseline).astype(np.float32)
    return corrected, baseline

