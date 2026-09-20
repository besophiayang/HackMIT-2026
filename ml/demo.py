"""Capture one touch from the configured source and classify it."""

import argparse

from .config import (
    LOCATIONS,
    HARDWARE_NUM_SAMPLES,
    HARDWARE_NUM_SENSORS,
    NUM_SAMPLES,
    NUM_SENSORS,
    SERIAL_BAUD,
    SERIAL_PORT,
    TOUCH_TYPES,
    USE_HARDWARE,
)
from .inference import TouchClassifier
from .source_base import TouchSource
from .source_serial import SerialTouchSource
from .source_simulated import SimulatedTouchSource


def create_source() -> TouchSource:
    if USE_HARDWARE:
        return SerialTouchSource(
            SERIAL_PORT, SERIAL_BAUD, HARDWARE_NUM_SENSORS, HARDWARE_NUM_SAMPLES
        )
    return SimulatedTouchSource()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--touch-type", choices=TOUCH_TYPES)
    parser.add_argument("--location", choices=LOCATIONS)
    args = parser.parse_args()

    source = create_source()
    if USE_HARDWARE:
        print("Waiting for one touch window from the Teensy...")
        waveform = source.get_touch()
    else:
        touch_type = args.touch_type or input(f"Touch type {TOUCH_TYPES}: ").strip()
        location = args.location or input(f"Location {LOCATIONS}: ").strip()
        waveform = source.get_touch(touch_type=touch_type, location=location)

    result = TouchClassifier(source="real" if USE_HARDWARE else "simulated").predict(waveform)
    print("\nDetected touch")
    print("-------------------")
    print(f"Type: {result['touch_type']}")
    print(f"Confidence: {result['touch_confidence']:.1%}")
    print(f"\nLocation: {result['location']}")
    print(f"Confidence: {result['location_confidence']:.1%}")
    print(f"\nPrediction dictionary: {result}")


if __name__ == "__main__":
    main()
