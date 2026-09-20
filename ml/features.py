"""Physics-informed time, frequency, and sensor-fusion features."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, hilbert, welch

from .config import SAMPLE_RATE
from .preprocessing import subtract_baseline
from .source_base import TouchSource

PER_SENSOR_FEATURES = [
    "peak_abs", "rms", "log_energy", "peak_to_peak", "time_to_peak",
    "dominant_frequency", "spectral_centroid", "spectral_bandwidth",
    "spectral_rolloff_85", "spectral_entropy", "crest_factor",
    "mean_absolute", "standard_deviation", "zero_crossing_rate",
    "impulse_count", "envelope_mean", "envelope_std", "envelope_peak_time",
    "duration_20pct", "early_late_energy_ratio", "periodicity",
]
ENERGY_BANDS = ((0, 40), (40, 100), (100, 200), (200, 350), (350, 500))
TEMPORAL_BINS = 6


def feature_names(num_sensors: int) -> list[str]:
    names = [f"sensor_{sensor}_{feature}" for sensor in range(num_sensors) for feature in PER_SENSOR_FEATURES]
    names += [f"sensor_{sensor}_band_{low}_{high}_relative_energy" for sensor in range(num_sensors) for low, high in ENERGY_BANDS]
    names += [f"sensor_{sensor}_relative_energy" for sensor in range(num_sensors)]
    names += [f"sensor_{sensor}_peak_time_difference" for sensor in range(num_sensors)]
    names += [f"sensor_{sensor}_time_bin_{time_bin}_energy" for sensor in range(num_sensors) for time_bin in range(TEMPORAL_BINS)]
    names += [name for left in range(num_sensors) for right in range(left + 1, num_sensors) for name in (f"corr_{left}_{right}", f"lag_{left}_{right}")]
    names += ["strongest_sensor_normalized", "strongest_energy_margin", "sensor_energy_entropy", "active_sensor_fraction", "global_impulse_count", "global_duration", "energy_motion"]
    return names


def _spectral_features(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, ...]:
    frequencies, power = welch(signal, fs=sample_rate, axis=0, nperseg=min(256, signal.shape[0]))
    power = np.maximum(power, 0)
    power_sum = np.sum(power, axis=0) + 1e-12
    dominant = frequencies[np.argmax(power, axis=0)]
    centroid = np.sum(frequencies[:, None] * power, axis=0) / power_sum
    bandwidth = np.sqrt(np.sum(((frequencies[:, None] - centroid) ** 2) * power, axis=0) / power_sum)
    cumulative = np.cumsum(power, axis=0)
    rolloff = np.asarray([frequencies[min(np.searchsorted(cumulative[:, sensor], 0.85 * power_sum[sensor]), len(frequencies) - 1)] for sensor in range(signal.shape[1])])
    probability = power / power_sum
    entropy = -np.sum(probability * np.log2(probability + 1e-12), axis=0) / np.log2(max(2, power.shape[0]))
    bands = []
    for low, high in ENERGY_BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        bands.append(np.sum(power[mask], axis=0) / power_sum)
    return dominant, centroid, bandwidth, rolloff, entropy, np.stack(bands, axis=1)


def extract_features(window: np.ndarray, sample_rate: int = SAMPLE_RATE, idle_samples: int | None = None) -> np.ndarray:
    """Convert one waveform to amplitude-robust, propagation-aware features."""
    TouchSource.validate_waveform(window)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    centered, _ = subtract_baseline(window, idle_samples)
    signal = centered.astype(np.float64)
    num_samples, num_sensors = signal.shape
    epsilon = 1e-12
    absolute = np.abs(signal)
    peak = np.max(absolute, axis=0)
    rms = np.sqrt(np.mean(signal**2, axis=0))
    energy = np.sum(signal**2, axis=0)
    peak_indices = np.argmax(absolute, axis=0)
    time_to_peak = peak_indices / sample_rate
    envelope = np.abs(hilbert(signal, axis=0))
    envelope_peak_indices = np.argmax(envelope, axis=0)
    envelope_threshold = 0.2 * np.max(envelope, axis=0)
    duration = np.sum(envelope >= envelope_threshold, axis=0) / sample_rate
    half = max(1, num_samples // 2)
    early_late_ratio = (np.sum(signal[:half] ** 2, axis=0) + epsilon) / (np.sum(signal[half:] ** 2, axis=0) + epsilon)
    zero_crossing = np.mean(np.diff(np.signbit(signal), axis=0), axis=0)

    impulse_counts, periodicity = [], []
    for sensor in range(num_sensors):
        scale = np.median(np.abs(signal[:, sensor] - np.median(signal[:, sensor]))) + epsilon
        peaks, _ = find_peaks(envelope[:, sensor], prominence=4 * scale, distance=max(1, sample_rate // 100))
        impulse_counts.append(len(peaks))
        autocorrelation = np.correlate(signal[:, sensor], signal[:, sensor], mode="full")[num_samples - 1:]
        autocorrelation /= autocorrelation[0] + epsilon
        start, stop = max(1, sample_rate // 500), min(len(autocorrelation), sample_rate // 20)
        periodicity.append(float(np.max(autocorrelation[start:stop])) if stop > start else 0.0)

    dominant, centroid, bandwidth, rolloff, spectral_entropy, band_energy = _spectral_features(signal, sample_rate)
    per_sensor = np.column_stack([
        peak, rms, np.log1p(energy), np.ptp(signal, axis=0), time_to_peak,
        dominant, centroid, bandwidth, rolloff, spectral_entropy,
        peak / (rms + epsilon), np.mean(absolute, axis=0), np.std(signal, axis=0),
        zero_crossing, impulse_counts, np.mean(envelope, axis=0), np.std(envelope, axis=0),
        envelope_peak_indices / sample_rate, duration, np.log1p(early_late_ratio), periodicity,
    ]).ravel()

    total_energy = np.sum(energy) + epsilon
    relative_energy = energy / total_energy
    strongest = int(np.argmax(energy))
    ordered = np.sort(relative_energy)
    margin = float(ordered[-1] - ordered[-2]) if num_sensors > 1 else float(ordered[-1])
    peak_differences = time_to_peak - time_to_peak[strongest]
    temporal_energy = []
    for sensor in range(num_sensors):
        chunks = np.array_split(signal[:, sensor] ** 2, TEMPORAL_BINS)
        temporal_energy.extend(float(np.sum(chunk) / (energy[sensor] + epsilon)) for chunk in chunks)

    pair_features = []
    max_lag = min(num_samples - 1, max(1, int(0.015 * sample_rate)))
    normalized = signal / (np.std(signal, axis=0, keepdims=True) + epsilon)
    for left in range(num_sensors):
        for right in range(left + 1, num_sensors):
            pair_features.append(float(np.mean(normalized[:, left] * normalized[:, right])))
            correlation = np.correlate(normalized[:, left], normalized[:, right], mode="full")
            middle = num_samples - 1
            local = correlation[middle - max_lag:middle + max_lag + 1]
            pair_features.append(float((np.argmax(local) - max_lag) / sample_rate))

    global_envelope = np.sqrt(np.sum(signal**2, axis=1))
    global_scale = np.median(np.abs(global_envelope - np.median(global_envelope))) + epsilon
    global_peaks, _ = find_peaks(global_envelope, prominence=4 * global_scale, distance=max(1, sample_rate // 100))
    global_duration = float(np.sum(global_envelope >= 0.2 * np.max(global_envelope)) / sample_rate)
    sensor_entropy = float(-np.sum(relative_energy * np.log2(relative_energy + epsilon)) / np.log2(max(2, num_sensors)))
    active_fraction = float(np.mean(relative_energy > (0.5 / num_sensors)))
    bin_matrix = np.asarray(temporal_energy).reshape(num_sensors, TEMPORAL_BINS)
    centroids = np.sum(bin_matrix * np.arange(num_sensors)[:, None], axis=0) / (np.sum(bin_matrix, axis=0) + epsilon)
    energy_motion = float(np.mean(np.abs(np.diff(centroids))) / max(1, num_sensors - 1))
    aggregate = np.asarray([strongest / max(1, num_sensors - 1), margin, sensor_entropy, active_fraction, len(global_peaks), global_duration, energy_motion])
    result = np.concatenate([per_sensor, band_energy.ravel(), relative_energy, peak_differences, temporal_energy, pair_features, aggregate]).astype(np.float32)
    expected = len(feature_names(num_sensors))
    if result.size != expected:
        raise RuntimeError(f"Feature schema mismatch: produced {result.size}, expected {expected}.")
    return result
