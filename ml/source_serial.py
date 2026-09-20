"""ESP32-S3 USB serial implementation of the touch-source interface."""

from collections import deque
import re
import time

import numpy as np

from .source_base import TouchSource


class SerialTouchSource(TouchSource):
    """Read ESP32 ``Sensor1:123,Sensor2:45,...`` samples.

    The older ``timestamp,s0,...`` format remains accepted for compatibility.
    """

    _LABELED_VALUE = re.compile(r"^Sensor(\d+):\s*(-?(?:\d+(?:\.\d*)?|\.\d+))$")

    def __init__(
        self, port: str, baud: int, num_sensors: int, num_samples: int
    ) -> None:
        # Import lazily so simulation and training do not require serial hardware.
        try:
            import serial
        except ImportError as exc:
            raise ImportError(
                "pyserial is required only for hardware mode. Run: pip install pyserial"
            ) from exc

        self.num_sensors = num_sensors
        self.num_samples = num_samples
        self._serial_module = serial
        self._read_buffer = bytearray()
        try:
            self.serial = serial.Serial(port=port, baudrate=baud, timeout=1.0)
        except serial.SerialException as exc:
            raise ConnectionError(f"Could not open serial port {port}: {exc}") from exc

    def parse_line(self, line: str) -> np.ndarray | None:
        """Parse one Arduino message without raising on malformed input."""
        fields = [field.strip() for field in line.strip().split(",")]
        labeled: dict[int, float] = {}
        for field in fields:
            match = self._LABELED_VALUE.match(field)
            if match:
                labeled[int(match.group(1))] = float(match.group(2))
        if len(labeled) == self.num_sensors:
            expected = list(range(1, self.num_sensors + 1))
            if sorted(labeled) == expected:
                values = np.asarray([labeled[index] for index in expected], dtype=np.float32)
                return values if np.isfinite(values).all() else None

        # Compatibility with timestamp,s0,s1,... from the original prototype.
        if len(fields) == self.num_sensors + 1:
            try:
                float(fields[0])
                values = np.asarray([float(value) for value in fields[1:]], dtype=np.float32)
                return values if np.isfinite(values).all() else None
            except ValueError:
                pass
        return None

    def read_sample(self, max_wait_seconds: float | None = None) -> np.ndarray:
        """Skip malformed lines until one valid sensor row is available."""
        started = time.monotonic()
        last_nonempty_line = ""
        while True:
            try:
                # Read available USB chunks, not one operating-system read per
                # byte (pyserial's generic readline path). Preserve partial rows.
                while b"\n" not in self._read_buffer:
                    self._read_buffer.extend(self.serial.read(max(1, self.serial.in_waiting)))
                    if len(self._read_buffer) > 1_000_000:
                        self._read_buffer.clear()
                        raise ValueError("Serial input has no line endings.")
                    if max_wait_seconds is not None and time.monotonic() - started >= max_wait_seconds:
                        raise TimeoutError("No complete sensor line received; check the ESP32 connection.")
                raw, _, remaining = self._read_buffer.partition(b"\n")
                self._read_buffer = bytearray(remaining)
                line = raw.decode("utf-8", errors="ignore").strip()
                if line:
                    last_nonempty_line = line
                values = self.parse_line(line)
                if values is not None:
                    return values
                if max_wait_seconds is not None and time.monotonic() - started >= max_wait_seconds:
                    detail = (
                        f" Last line received: {last_nonempty_line!r}."
                        if last_nonempty_line
                        else " No serial text was received."
                    )
                    raise TimeoutError(
                        f"No valid {self.num_sensors}-sensor line arrived within "
                        f"{max_wait_seconds:g} seconds.{detail} Check the uploaded "
                        "firmware, COM port, baud rate, and close Serial Monitor."
                    )
            except self._serial_module.SerialException as exc:
                raise ConnectionError(f"Serial connection lost: {exc}") from exc
            except (ValueError, TypeError):
                continue

    def get_touch(self) -> np.ndarray:
        rows: list[np.ndarray] = []
        while len(rows) < self.num_samples:
            rows.append(self.read_sample())
        waveform = np.asarray(rows, dtype=np.float32)
        return self.validate_waveform(waveform, self.num_samples, self.num_sensors)

    def get_triggered_touch(
        self,
        threshold: float,
        baseline_samples: int = 100,
        pretrigger_samples: int = 50,
        on_armed: object | None = None,
        noise_multiplier: float = 6.0,
    ) -> np.ndarray:
        """Wait for a transient above the robot's learned vibration baseline.

        ``threshold`` is a minimum ADC excursion *after* the normal per-channel
        vibration envelope has been removed. ``noise_multiplier`` controls how
        many robust noise scales a sample must exceed.
        """
        if threshold <= 0:
            raise ValueError("threshold must be positive.")
        if baseline_samples < 2 or not 1 <= pretrigger_samples < self.num_samples:
            raise ValueError("Invalid baseline/pre-trigger sample count.")
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive.")

        history: deque[np.ndarray] = deque(maxlen=baseline_samples)
        pretrigger: deque[np.ndarray] = deque(maxlen=pretrigger_samples)
        recent_scores: deque[float] = deque(maxlen=5)
        armed_announced = False
        while True:
            sample = self.read_sample()
            if len(history) >= baseline_samples:
                if not armed_announced:
                    armed_announced = True
                    if callable(on_armed):
                        on_armed()
                history_array = np.stack(history)
                baseline = np.median(history_array, axis=0)
                # Median absolute deviation is robust to occasional bumps while
                # representing the normal motor/stance vibration on each piezo.
                mad = np.median(np.abs(history_array - baseline), axis=0)
                robust_noise = np.maximum(1.4826 * mad, 0.5)
                excess = np.abs(sample - baseline) - noise_multiplier * robust_noise
                score = float(np.max(excess))
                recent_scores.append(score)
                # A real piezo event normally contains an impulse plus ringing.
                # Requiring two elevated samples rejects isolated ADC glitches.
                sustained = sum(value >= threshold * 0.35 for value in recent_scores) >= 2
                if score >= threshold and sustained:
                    rows = list(pretrigger) + [sample]
                    while len(rows) < self.num_samples:
                        rows.append(self.read_sample())
                    waveform = np.asarray(rows[-self.num_samples :], dtype=np.float32)
                    return self.validate_waveform(
                        waveform, self.num_samples, self.num_sensors
                    )
            history.append(sample)
            pretrigger.append(sample)

    def close(self) -> None:
        self.serial.close()

    def reset_input_buffer(self) -> None:
        self._read_buffer.clear()
        self.serial.reset_input_buffer()

    def __enter__(self) -> "SerialTouchSource":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
