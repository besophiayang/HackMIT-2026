"""Varied, intentionally lightweight simulation of piezo touch signals."""

import numpy as np

from .config import LOCATIONS, NUM_SAMPLES, NUM_SENSORS, SAMPLE_RATE, TOUCH_TYPES
from .source_base import TouchSource


class SimulatedTouchSource(TouchSource):
    """Generate synthetic multi-sensor signals for pipeline development."""

    # Approximate sensor positions from nose (0) to tail (1), with paired sensors.
    _SENSOR_POSITIONS = np.array(
        [0.02, 0.08, 0.23, 0.28, 0.48, 0.58, 0.80, 0.96], dtype=np.float32
    )
    _LOCATION_POSITIONS = {
        "head": 0.04,
        "front_left": 0.22,
        "front_right": 0.30,
        "mid_back": 0.55,
        "rear": 0.90,
    }

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def _event_signal(self, touch_type: str, local_time: np.ndarray) -> np.ndarray:
        """Create the base vibration at one sensor after its arrival time."""
        active = local_time >= 0
        t = np.maximum(local_time, 0.0)

        if touch_type in {"tap", "hard_tap", "poke"}:
            hard = touch_type == "hard_tap"
            poke = touch_type == "poke"
            amplitude = self.rng.uniform(1.7, 2.3) if hard else self.rng.uniform(0.55, 0.9) if poke else self.rng.uniform(0.75, 1.25)
            frequency = self.rng.uniform(105, 195)
            decay = self.rng.uniform(70, 110) if poke else self.rng.uniform(24, 34) if hard else self.rng.uniform(32, 46)
            # A narrow impact plus decaying piezo ringing.
            impulse = np.exp(-0.5 * (t / self.rng.uniform(0.0007, 0.0015)) ** 2)
            ringing = np.exp(-decay * t) * np.sin(2 * np.pi * frequency * t)
            signal = amplitude * (0.8 * impulse + ringing)
        elif touch_type == "scratch":
            signal = np.zeros_like(t)
            frequency = self.rng.uniform(220, 480)
            count = int(self.rng.integers(5, 10))
            for event_time in np.linspace(0, self.rng.uniform(0.14, 0.23), count):
                event_time += self.rng.normal(0, 0.003)
                dt = t - event_time
                on = dt >= 0
                signal += (
                    on
                    * self.rng.uniform(0.35, 0.75)
                    * np.exp(-self.rng.uniform(55, 90) * np.maximum(dt, 0))
                    * np.sin(2 * np.pi * frequency * np.maximum(dt, 0))
                )
        else:  # pet or stroke
            frequency = self.rng.uniform(45, 110) if touch_type == "stroke" else self.rng.uniform(25, 75)
            duration = self.rng.uniform(0.16, 0.27)
            envelope = np.sin(np.pi * np.clip(t / duration, 0, 1)) ** 2
            envelope[t > duration] = 0
            amplitude = self.rng.uniform(0.35, 0.65) if touch_type == "stroke" else self.rng.uniform(0.22, 0.48)
            signal = amplitude * envelope * np.sin(
                2 * np.pi * frequency * t + self.rng.uniform(-0.3, 0.3)
            )

        return signal * active

    def get_touch(self, touch_type: str = "tap", location: str = "head") -> np.ndarray:
        if touch_type not in TOUCH_TYPES:
            raise ValueError(f"Unknown touch type {touch_type!r}; choose from {TOUCH_TYPES}.")
        if location not in LOCATIONS:
            raise ValueError(f"Unknown location {location!r}; choose from {LOCATIONS}.")

        time = np.arange(NUM_SAMPLES, dtype=np.float32) / SAMPLE_RATE
        touch_position = self._LOCATION_POSITIONS[location]
        start_time = self.rng.uniform(0.025, 0.065)
        waveform = np.empty((NUM_SAMPLES, NUM_SENSORS), dtype=np.float32)

        for sensor, sensor_position in enumerate(self._SENSOR_POSITIONS):
            distance = abs(float(sensor_position) - touch_position)
            if touch_type == "stroke":
                # A stroke moves its energy centroid across neighboring sensors.
                moving_position = touch_position + np.linspace(-0.14, 0.14, NUM_SAMPLES)
                attenuation = np.exp(-2.7 * np.abs(float(sensor_position) - moving_position))
                attenuation *= self.rng.uniform(0.86, 1.14)
            else:
                attenuation = np.exp(-2.7 * distance) * self.rng.uniform(0.86, 1.14)
            propagation_delay = distance * self.rng.uniform(0.004, 0.008)
            timing_jitter = self.rng.normal(0, 0.0007)
            local_time = time - start_time - propagation_delay - timing_jitter
            signal = attenuation * self._event_signal(touch_type, local_time)
            noise = self.rng.normal(0, 0.012, NUM_SAMPLES)
            slow_drift = self.rng.normal(0, 0.008) * np.linspace(-1, 1, NUM_SAMPLES)
            waveform[:, sensor] = signal + noise + slow_drift

        waveform = waveform.astype(np.float32)
        return self.validate_waveform(waveform)
