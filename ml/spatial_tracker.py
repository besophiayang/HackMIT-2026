"""Low-latency spatial fingerprint tracker for continuous touch motion."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

import joblib
import numpy as np
import pandas as pd

from .body_map import BodySurfaceMap
from .config import LABELS_PATH, MODELS_DIR, RAW_DATA_DIR
from .motion_mapping import spatial_vector
from .preprocessing import subtract_baseline
from .surface_calibration import SurfaceCalibration


@dataclass
class SpatialEstimate:
    u: float
    v: float
    confidence: float
    location: str
    probabilities: dict[str, float]
    side: str | None


class SpatialFingerprintTracker:
    """Match short-term sensor energy ratios to measured location templates."""

    def __init__(
        self,
        names: list[str],
        templates: np.ndarray,
        sensor_scales: np.ndarray,
        amplitude_low: float,
        amplitude_high: float,
        body_map: BodySurfaceMap,
        motion_model: dict[str, object] | None = None,
    ) -> None:
        self.names = names
        self.templates = templates.astype(np.float64)
        self.sensor_scales = sensor_scales.astype(np.float64)
        self.amplitude_low = float(amplitude_low)
        self.amplitude_high = float(amplitude_high)
        self.motion_model = motion_model
        self.placements = {placement.name: placement for placement in body_map.placements}
        self.by_channel = sorted(body_map.placements, key=lambda placement: placement.channel)
        self.surface = SurfaceCalibration(self.sensor_scales)
        self._locked_side: str | None = None
        self._side_candidate: str | None = None
        self._side_candidate_count = 0
        self._motion_history: deque[np.ndarray] = deque(maxlen=60)
        self._motion_update_count = 0
        self._cached_motion: tuple[float, str, float, complex] | None = None
        self._motion_executor = None
        self._motion_future = None
        self._generation = 0

    def enable_async_motion(self) -> None:
        """Live heat never waits for the larger temporal forests."""
        self._motion_executor = ThreadPoolExecutor(max_workers=1)

    def close(self) -> None:
        if self._motion_executor:
            self._motion_executor.shutdown(wait=False, cancel_futures=True)

    def _predict_motion(self, sequence):
        motion_u = float(np.clip(self.motion_model['u_model'].predict(sequence)[0], 0, 1))
        probabilities = self.motion_model['side_model'].predict_proba(sequence)[0]
        by_side = dict(zip(self.motion_model['side_model'].classes_, probabilities, strict=True))
        side = max(by_side, key=by_side.get)
        return motion_u, side, float(by_side[side]), complex(0, float(by_side.get('right', 0)-by_side.get('left', 0)))

    def reset(self) -> None:
        """Forget temporal side state between separate touch events."""
        self._locked_side = None
        self._side_candidate = None
        self._side_candidate_count = 0
        self._motion_history.clear()
        self._motion_update_count = 0
        self._cached_motion = None
        self._generation += 1

    def _stable_side(self, values: np.ndarray) -> str | None:
        left = sum(values[p.channel] for p in self.by_channel if p.v >= 0.55)
        right = sum(values[p.channel] for p in self.by_channel if 0.08 <= p.v <= 0.42)
        top = sum(values[p.channel] for p in self.by_channel if p.v < 0.08 or p.v > 0.92)
        candidate: str | None = None
        if left > right * 1.05 and left > top * 0.72:
            candidate = "left"
        elif right > left * 1.05 and right > top * 0.72:
            candidate = "right"
        elif top > max(left, right) * 1.20:
            candidate = "top"

        return self._commit_side(candidate)

    def _commit_side(self, candidate: str | None) -> str | None:
        """Apply persistence before accepting a side change."""
        if candidate == self._locked_side:
            self._side_candidate = None
            self._side_candidate_count = 0
            return self._locked_side
        if candidate != self._side_candidate:
            self._side_candidate = candidate
            self._side_candidate_count = 1
        else:
            self._side_candidate_count += 1
        # Lock immediately at touch onset, then require persistent contrary
        # evidence before crossing the torso during one continuous stroke.
        required = 1 if self._locked_side is None else 6
        if candidate is not None and self._side_candidate_count >= required:
            self._locked_side = candidate
            self._side_candidate = None
            self._side_candidate_count = 0
        return self._locked_side

    @classmethod
    def from_real_dataset(cls, sensor_count: int, body_map: BodySurfaceMap) -> "SpatialFingerprintTracker":
        labels = pd.read_csv(LABELS_PATH)
        rows = labels[(labels["source"] == "real") & (labels["num_sensors"] == sensor_count)]
        known = {placement.name for placement in body_map.placements}
        rows = rows[rows["location"].isin(known)]
        if rows.empty:
            raise ValueError("No matching five-sensor location recordings were found.")
        responses: dict[str, list[np.ndarray]] = {name: [] for name in known}
        for row in rows.itertuples(index=False):
            waveform = np.load(RAW_DATA_DIR / row.file, allow_pickle=False)
            centered, _ = subtract_baseline(waveform)
            rms = np.sqrt(np.mean(centered.astype(np.float64) ** 2, axis=0))
            if np.linalg.norm(rms) > 1e-8:
                responses[str(row.location)].append(rms)

        # Piezo mounting and analog tolerances create large channel-gain
        # differences. Estimate each channel's gain from touches labeled at that
        # physical sensor, then express all live/template energy in comparable units.
        sensor_scales = np.ones(sensor_count, dtype=np.float64)
        for placement in body_map.placements:
            samples = responses[placement.name]
            if samples:
                sensor_scales[placement.channel] = np.median(
                    [sample[placement.channel] for sample in samples]
                )
        valid_scales = sensor_scales[sensor_scales > 1e-8]
        sensor_scales /= np.median(valid_scales) if valid_scales.size else 1.0
        # Existing examples were not recorded at controlled equal forces. Treat
        # their medians as a weak gain estimate, not permission to amplify a
        # quiet channel arbitrarily (GPIO 15 previously received >2x gain).
        sensor_scales = np.clip(np.sqrt(sensor_scales), 0.75, 1.5)

        calibrated_amplitudes = [
            float(np.max(sample / sensor_scales))
            for samples in responses.values()
            for sample in samples
        ]
        amplitude_low, amplitude_high = np.percentile(calibrated_amplitudes, [5, 90])

        names, templates = [], []
        for placement in body_map.placements:
            samples = responses[placement.name]
            if not samples:
                continue
            normalized = np.stack([sample / sensor_scales for sample in samples])
            normalized /= np.linalg.norm(normalized, axis=1, keepdims=True) + 1e-12
            template = np.median(normalized, axis=0)
            template /= np.linalg.norm(template) + 1e-12
            names.append(placement.name)
            templates.append(template)
        if len(names) < 2:
            raise ValueError("At least two trained location fingerprints are required.")
        motion_model: dict[str, object] | None = None
        motion_path = MODELS_DIR / "spatial_motion_real.pkl"
        if motion_path.exists():
            artifact = joblib.load(motion_path)
            if (
                int(artifact.get("sensor_count", -1)) == sensor_count
                and int(artifact.get("feature_schema", -1)) == 2
            ):
                motion_model = artifact
                for model in (artifact["u_model"], artifact["side_model"]):
                    if hasattr(model, "n_jobs"):
                        model.n_jobs = 1
        return cls(
            names, np.stack(templates), sensor_scales,
            amplitude_low, amplitude_high, body_map, motion_model,
        )

    def characterize(self, strengths: np.ndarray) -> tuple[float, float]:
        """Return data-calibrated contact intensity and spatial area in [0, 1]."""
        values = np.maximum(np.asarray(strengths, dtype=np.float64), 0) / self.sensor_scales
        level = float(np.max(values))
        low = 0.0
        high = np.log1p(max(self.amplitude_high, self.amplitude_low + 1e-6))
        intensity = float(np.clip((np.log1p(level) - low) / (high - low + 1e-12), 0.0, 1.0))

        # Participation ratio distinguishes one compact responding sensor from
        # a broader multi-sensor contact. Ignore tiny propagated vibration first.
        if level <= 1e-12:
            return intensity, 0.0
        weights = np.where(values >= level * 0.12, values, 0.0)
        weights /= np.sum(weights) + 1e-12
        effective_sensors = 1.0 / (np.sum(weights**2) + 1e-12)
        area = float(np.clip((effective_sensors - 1.0) / max(len(values) - 1, 1), 0.0, 1.0))
        return intensity, area

    def estimate(self, strengths: np.ndarray) -> SpatialEstimate | None:
        values = np.maximum(np.asarray(strengths, dtype=np.float64), 0) / self.sensor_scales
        norm = np.linalg.norm(values)
        if norm <= 1e-8:
            return None
        fingerprint = values / norm
        self._motion_history.append(spatial_vector(strengths, self.sensor_scales))
        similarity = np.clip(self.templates @ fingerprint, -1, 1)
        # Softmax over cosine similarities produces smooth interpolation while
        # emphasizing the measured template that best explains the response.
        logits = 18.0 * (similarity - np.max(similarity))
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities)
        learned_u = 0.0
        learned_circular = 0j
        for name, probability in zip(self.names, probabilities, strict=True):
            placement = self.placements[name]
            learned_u += float(probability) * placement.u
            learned_circular += float(probability) * np.exp(2j * np.pi * placement.v)

        # A gain-corrected centroid guarantees that energy moving between nearby
        # physical sensors moves in the same direction on the robot. Weak common
        # vibration is suppressed before the centroid so it cannot pull every
        # estimate toward the middle of the body.
        relative = values / (np.max(values) + 1e-12)
        direct_weights = np.maximum(relative - 0.04, 0.0)
        direct_weights /= np.sum(direct_weights) + 1e-12
        direct_u = sum(weight * placement.u for weight, placement in zip(direct_weights, self.by_channel, strict=True))
        direct_circular = sum(
            weight * np.exp(2j * np.pi * placement.v)
            for weight, placement in zip(direct_weights, self.by_channel, strict=True)
        )
        u = 0.85 * direct_u + 0.15 * learned_u
        circular = 0.85 * direct_circular + 0.15 * learned_circular
        motion_side: str | None = None
        motion_side_confidence = 0.0
        history_array = np.stack(self._motion_history)
        motion_variation = (
            float(np.mean(np.linalg.norm(np.diff(history_array[-12:], axis=0), axis=1)))
            if len(history_array) >= 8 else 0.0
        )
        is_moving = motion_variation >= 0.02
        # The old motion artifact stores features normalized with older gains.
        # Only artifacts explicitly carrying matching scales may influence heat.
        compatible_motion = self.motion_model is not None and np.allclose(
            self.motion_model.get('sensor_scales', np.zeros_like(self.sensor_scales)), self.sensor_scales
        )
        if compatible_motion and is_moving:
            self._motion_update_count += 1
            history = list(self._motion_history)
            if self._cached_motion is None or self._motion_update_count % 5 == 0:
                padded = history
                if len(padded) < 4:
                    padded = [padded[0]] * (4 - len(padded)) + padded
                temporal_bins = [np.mean(chunk, axis=0) for chunk in np.array_split(padded, 4)]
                sequence = np.concatenate([history[-1], *temporal_bins]).reshape(1, -1)
                if self._motion_executor is None:
                    self._cached_motion = self._predict_motion(sequence)
                else:
                    if self._motion_future is not None and self._motion_future.done():
                        generation, started = self._motion_job
                        result = self._motion_future.result()
                        if generation == self._generation and time.monotonic()-started < 0.12:
                            self._cached_motion = result
                        self._motion_future = None
                    if self._motion_future is None:
                        self._motion_job = (self._generation, time.monotonic())
                        self._motion_future = self._motion_executor.submit(self._predict_motion, sequence)
            if self._cached_motion is not None:
                motion_u, motion_side, motion_side_confidence, motion_circular = self._cached_motion
                # Weakly labelled motion is a prior, never a replacement for
                # current physical sensor evidence or a compulsory visual delay.
                u = 0.20 * motion_u + 0.65 * direct_u + 0.15 * learned_u
                circular = 0.20 * motion_circular + 0.65 * direct_circular + 0.15 * learned_circular
        elif not is_moving:
            self._cached_motion = None
        left_energy = sum(values[p.channel] for p in self.by_channel if p.v >= 0.55)
        right_energy = sum(values[p.channel] for p in self.by_channel if 0.08 <= p.v <= 0.42)
        top_energy = sum(values[p.channel] for p in self.by_channel if p.v < 0.08 or p.v > 0.92)
        top_dominant = top_energy > max(left_energy, right_energy) * 1.35
        if (
            motion_side is not None
            and motion_side_confidence >= 0.58
            and len(self._motion_history) >= 8
            and not top_dominant
        ):
            stable_side = self._commit_side(motion_side)
        else:
            stable_side = self._stable_side(values)
        side_v = {"left": 0.75, "right": 0.25, "top": 0.0}.get(stable_side)
        if side_v is not None:
            # Strong temporal anchor prevents propagated vibration from painting
            # the opposite side during one continuous pet, while four consistent
            # updates still allow a real movement around the body to cross sides.
            # A mild temporal prior stabilizes side evidence while retaining
            # real intermediate positions between a side panel and the top.
            circular = 0.8 * circular + 0.2 * np.exp(2j * np.pi * side_v)
        v = float((np.angle(circular) / (2 * np.pi)) % 1.0)
        measured = self.surface.estimate(values)
        if measured is not None:
            u, v = measured
        # A truly isolated channel is an exact physical anchor. In particular,
        # GPIO 6 must not be dragged toward GPIO 7's correlated template.
        if float(np.max(values) / (np.sum(values) + 1e-12)) > 0.95:
            anchor = self.by_channel[int(np.argmax(values))]
            u, v = anchor.u, anchor.v
            stable_side = 'top' if anchor.v < .08 or anchor.v > .92 else ('left' if anchor.v > .5 else 'right')
        # The torso surface is cylindrical: mirror an ambiguous opposite-side
        # estimate back onto the side supported by the current event. This
        # preserves height/top transitions while forbidding cross-body heat.
        if stable_side == 'left' and v < .5:
            v = 1.0-v
        elif stable_side == 'right' and v > .5:
            v = 1.0-v
        distance = [(p.u-u)**2 + (((p.v-v+0.5)%1)-0.5)**2 for p in self.by_channel]
        nearest = self.by_channel[int(np.argmin(distance))].name
        return SpatialEstimate(
            u=float(u),
            v=v,
            confidence=float(np.max(probabilities)),
            location=nearest,
            probabilities={name: float(value) for name, value in zip(self.names, probabilities, strict=True)},
            side=stable_side,
        )
