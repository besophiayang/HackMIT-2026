"""Map five piezo strengths onto a flattened Unitree Go2 torso surface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .config import DATA_DIR, HARDWARE_SENSOR_PINS
from .preprocessing import subtract_baseline
from .source_base import TouchSource

SENSOR_MAP_PATH = DATA_DIR / "sensor_map.json"


@dataclass(frozen=True)
class SensorPlacement:
    """One sensor location on the flattened body net.

    ``u`` runs from head/front (0) to tail/rear (1). ``v`` wraps around the
    torso: top=0/1, robot-right=0.25, underside=0.5, robot-left=0.75.
    """

    channel: int
    pin: int
    name: str
    u: float
    v: float


# Approximate red-circle positions from the supplied two views. Calibration
# associates the actual GPIO channel with each physical point.
CALIBRATION_TARGETS = [
    {"name": "front_left_side", "u": 0.24, "v": 0.75},
    {"name": "rear_left_side", "u": 0.80, "v": 0.75},
    {"name": "front_right_side", "u": 0.24, "v": 0.25},
    {"name": "rear_right_side", "u": 0.80, "v": 0.25},
    {"name": "rear_top_butt", "u": 0.82, "v": 0.02},
]

# Sensor 5 is physically mounted on the top-rear/butt panel. Keep this
# hardware fact fixed during calibration; the other four channels are inferred
# from their measured response matrix.
FIXED_CHANNEL_TARGETS = {
    0: "rear_left_side",   # GPIO 4
    1: "rear_right_side",  # GPIO 5
    2: "rear_top_butt",    # GPIO 6
    3: "front_right_side", # GPIO 7
    4: "front_left_side",  # GPIO 15
}


class BodySurfaceMap:
    """Estimate a continuous touch coordinate from distributed piezos."""

    def __init__(self, placements: list[SensorPlacement]) -> None:
        if len(placements) < 2:
            raise ValueError("At least two sensor placements are required.")
        channels = sorted(item.channel for item in placements)
        if channels != list(range(len(placements))):
            raise ValueError("Sensor channels must be unique and zero-based.")
        if any(not 0 <= item.u <= 1 or not 0 <= item.v <= 1 for item in placements):
            raise ValueError("Sensor coordinates u and v must be between 0 and 1.")
        self.placements = sorted(placements, key=lambda item: item.channel)

    @classmethod
    def provisional(cls) -> "BodySurfaceMap":
        """Return the known fixed GPIO-to-body installation map."""
        targets = {str(target["name"]): target for target in CALIBRATION_TARGETS}
        placements = []
        for channel, pin in enumerate(HARDWARE_SENSOR_PINS):
            target = targets[FIXED_CHANNEL_TARGETS[channel]]
            placements.append(
                SensorPlacement(
                    channel, pin, str(target["name"]), float(target["u"]), float(target["v"])
                )
            )
        return cls(placements)

    @classmethod
    def load(cls, path: Path = SENSOR_MAP_PATH) -> "BodySurfaceMap":
        if not path.exists():
            return cls.provisional()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls([SensorPlacement(**item) for item in payload["placements"]])

    def save(self, path: Path = SENSOR_MAP_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "coordinate_system": {
                "u": "0=head/front, 1=tail/rear",
                "v": "0/1=top, 0.25=right, 0.5=underside, 0.75=left",
            },
            "placements": [asdict(item) for item in self.placements],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def localize_strengths(self, strengths: np.ndarray) -> dict[str, object]:
        """Calculate a continuous point using a weighted circular centroid."""
        values = np.asarray(strengths, dtype=np.float64)
        if values.shape != (len(self.placements),):
            raise ValueError(f"Expected {len(self.placements)} sensor strengths.")
        values = np.maximum(values, 0)
        total = float(np.sum(values))
        if total <= 1e-8:
            return {
                "detected": False,
                "u": None,
                "v": None,
                "confidence": 0.0,
                "strongest_sensor": None,
                "contributions": [0.0] * len(values),
            }

        # Slightly emphasize the strongest channels while retaining smooth
        # interpolation between sensors.
        weights = np.power(values, 1.5)
        weights /= np.sum(weights)
        u_positions = np.asarray([item.u for item in self.placements])
        v_positions = np.asarray([item.v for item in self.placements])
        u = float(np.sum(weights * u_positions))
        angles = 2 * np.pi * v_positions
        vector = np.sum(weights * np.exp(1j * angles))
        v = float((np.angle(vector) / (2 * np.pi)) % 1.0)
        strongest = int(np.argmax(values))
        concentration = float(np.abs(vector))
        dominance = float(weights[strongest])
        confidence = float(np.clip(0.5 * concentration + 0.5 * dominance, 0, 1))
        return {
            "detected": True,
            "u": u,
            "v": v,
            "confidence": confidence,
            "strongest_sensor": self.placements[strongest].name,
            "contributions": [float(value) for value in weights],
        }

    def localize_window(
        self, waveform: np.ndarray, idle_samples: int | None = None
    ) -> dict[str, object]:
        """Estimate body position from a raw ``(time, sensors)`` window."""
        TouchSource.validate_waveform(
            waveform, num_sensors=len(self.placements)
        )
        corrected, baseline = subtract_baseline(waveform, idle_samples)
        strengths = np.sqrt(np.mean(corrected.astype(np.float64) ** 2, axis=0))
        result = self.localize_strengths(strengths)
        result["strengths"] = [float(value) for value in strengths]
        result["baseline"] = [float(value) for value in baseline]
        return result

    @staticmethod
    def region_name(u: float, v: float) -> str:
        longitudinal = "front" if u < 0.34 else "middle" if u < 0.67 else "rear"
        if v < 0.125 or v >= 0.875:
            surface = "top"
        elif v < 0.375:
            surface = "right side"
        elif v < 0.625:
            surface = "underside"
        else:
            surface = "left side"
        return f"{longitudinal} {surface}"
