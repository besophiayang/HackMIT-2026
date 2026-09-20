"""Capture labeled multi-sensor ESP32 touch windows into the shared dataset."""

import argparse

from .config import (
    HARDWARE_NUM_SAMPLES,
    HARDWARE_NUM_SENSORS,
    HARDWARE_SAMPLE_RATE,
    SERIAL_BAUD,
)
from .source_serial import SerialTouchSource
from .storage import save_touch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--touch-type", required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--threshold", required=True, type=float, help="ADC deviation that starts capture")
    parser.add_argument("--count", type=int, default=1, help="Number of labeled examples to capture")
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD)
    parser.add_argument("--baseline-samples", type=int, default=100)
    parser.add_argument("--pretrigger-samples", type=int, default=50)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")

    source = SerialTouchSource(
        args.port, args.baud, HARDWARE_NUM_SENSORS, HARDWARE_NUM_SAMPLES
    )
    print("Keep the robot still while the baseline fills, then perform the labeled touch.")
    print("Press Ctrl+C to stop without losing examples already saved.")
    try:
        for index in range(args.count):
            print(f"Waiting for touch {index + 1}/{args.count}...")
            waveform = source.get_triggered_touch(
                args.threshold, args.baseline_samples, args.pretrigger_samples
            )
            path = save_touch(
                waveform,
                args.touch_type,
                args.location,
                source="real",
                sample_rate=HARDWARE_SAMPLE_RATE,
            )
            print(f"Saved {path.name} with shape {waveform.shape}")
    except KeyboardInterrupt:
        print("\nCapture stopped.")
    finally:
        source.close()


if __name__ == "__main__":
    main()
