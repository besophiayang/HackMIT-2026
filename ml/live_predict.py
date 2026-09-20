"""Detect touches from the ESP32 and classify them with real-data models."""

import argparse

from .body_map import BodySurfaceMap
from .config import (
    HARDWARE_NUM_SAMPLES,
    HARDWARE_NUM_SENSORS,
    SERIAL_BAUD,
)
from .inference import TouchClassifier
from .source_serial import SerialTouchSource


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--threshold", required=True, type=float, help="ADC deviation that indicates a touch")
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD)
    parser.add_argument("--baseline-samples", type=int, default=100)
    parser.add_argument("--pretrigger-samples", type=int, default=50)
    args = parser.parse_args()

    classifier = TouchClassifier(source="real")
    body_map = BodySurfaceMap.load()
    source = SerialTouchSource(
        args.port, args.baud, HARDWARE_NUM_SENSORS, HARDWARE_NUM_SAMPLES
    )
    print("Watching for touches. Press Ctrl+C to stop.")
    try:
        while True:
            waveform = source.get_triggered_touch(
                args.threshold, args.baseline_samples, args.pretrigger_samples
            )
            result = classifier.predict(waveform)
            surface = body_map.localize_window(waveform)
            print("\nTOUCH DETECTED")
            print(f"Type: {result['touch_type']}")
            print(f"Type confidence: {result['touch_confidence']:.1%}")
            print(f"Location: {result['location']}")
            print(f"Location confidence: {result['location_confidence']:.1%}")
            if surface["detected"]:
                u, v = float(surface["u"]), float(surface["v"])
                print(f"Surface point: {body_map.region_name(u, v)} (u={u:.2f}, v={v:.2f})")
                print(f"Surface confidence: {float(surface['confidence']):.1%}")
    except KeyboardInterrupt:
        print("\nLive prediction stopped.")
    finally:
        source.close()


if __name__ == "__main__":
    main()
