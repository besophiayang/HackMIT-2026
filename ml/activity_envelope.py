"""Fast per-channel activity above the powered robot's idle vibration envelope."""
from __future__ import annotations
import numpy as np


def window_rms(window: np.ndarray) -> np.ndarray:
    values=np.asarray(window,dtype=np.float64)
    values-=np.median(values,axis=0,keepdims=True)
    return np.sqrt(np.mean(values**2,axis=0))


class ActivityEnvelope:
    """Estimate activity in ADC-RMS units without waiting for an ML decision."""
    def __init__(self, bootstrap: np.ndarray, sensor_scales: np.ndarray, window_size: int=12):
        values=np.asarray(bootstrap,dtype=np.float64)
        if len(values)<window_size*4:
            raise ValueError('Not enough untouched startup data for the vibration envelope.')
        windows=np.stack([window_rms(values[end-window_size:end])
                          for end in range(window_size,len(values)+1,4)])
        self.idle_rms=np.quantile(windows,.90,axis=0)
        self.sensor_scales=np.asarray(sensor_scales,dtype=np.float64)
        self.window_size=window_size

    def measure(self, samples: np.ndarray) -> tuple[np.ndarray,np.ndarray,float]:
        rms=window_rms(np.asarray(samples)[-self.window_size:])
        excess=np.maximum(rms-self.idle_rms,0.0)
        corrected=excess/np.maximum(self.sensor_scales,1e-6)
        return rms,excess,float(np.max(corrected))

