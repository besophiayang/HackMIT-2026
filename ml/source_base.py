"""Common interface shared by simulated and physical touch sources."""

from abc import ABC, abstractmethod

import numpy as np


class TouchSource(ABC):
    """Base class for anything that captures one touch waveform."""

    @abstractmethod
    def get_touch(self, *args, **kwargs) -> np.ndarray:
        """Return one float32 waveform with shape (samples, sensors)."""
        raise NotImplementedError

    @staticmethod
    def validate_waveform(
        waveform: np.ndarray,
        num_samples: int | None = None,
        num_sensors: int | None = None,
    ) -> np.ndarray:
        """Validate the data contract and return the unchanged waveform."""
        if not isinstance(waveform, np.ndarray):
            raise ValueError("Waveform must be a NumPy array.")
        if waveform.ndim != 2 or waveform.shape[0] < 2 or waveform.shape[1] < 1:
            raise ValueError(
                "Waveform must have shape (num_samples, num_sensors), with at "
                "least two samples and one sensor."
            )
        if num_samples is not None and waveform.shape[0] != num_samples:
            raise ValueError(
                f"Waveform has {waveform.shape[0]} samples; expected {num_samples}."
            )
        if num_sensors is not None and waveform.shape[1] != num_sensors:
            raise ValueError(
                f"Waveform has {waveform.shape[1]} sensors; expected {num_sensors}."
            )
        if waveform.dtype != np.float32:
            raise ValueError(
                f"Waveform has dtype {waveform.dtype}; expected np.float32."
            )
        if np.isnan(waveform).any():
            raise ValueError("Waveform contains NaN values.")
        if np.isinf(waveform).any():
            raise ValueError("Waveform contains infinite values.")
        return waveform
