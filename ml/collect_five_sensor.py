"""Guided balanced data collection for the calibrated five-piezo robot shell."""

from __future__ import annotations

import argparse

from .body_map import BodySurfaceMap
from .config import (
    HARDWARE_NUM_SAMPLES,
    HARDWARE_NUM_SENSORS,
    HARDWARE_SAMPLE_RATE,
    SERIAL_BAUD,
    TOUCH_TYPES,
)
from .source_serial import SerialTouchSource
from .storage import save_touch


GESTURE_INSTRUCTIONS = {
    "tap": "one quick, light fingertip tap directly over the target sensor",
    "hard_tap": "one clearly firmer fingertip tap; do not hit hard enough to damage the shell",
    "poke": "one short fingertip press-and-release directly over the target (piezo senses the transitions)",
    "scratch": "3–5 short fingernail scratches in a small patch centered on the target",
    "stroke": "one deliberate 10–15 cm finger stroke that passes across the target",
    "pet": "one slow, broad palm/finger petting motion centered on the target area",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--count", type=int, default=20, help="Examples per gesture at each sensor")
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD)
    parser.add_argument("--gestures", nargs="+", choices=TOUCH_TYPES, default=TOUCH_TYPES)
    args = parser.parse_args()
    if args.count < 5:
        parser.error("Use at least 5 examples per gesture/location; 20–30 is recommended.")

    body_map = BodySurfaceMap.load()
    if len(body_map.placements) != HARDWARE_NUM_SENSORS:
        raise ValueError("The calibration map does not contain all five sensors.")

    total = len(args.gestures) * len(body_map.placements) * args.count
    print(f"Will collect {total} five-sensor windows.")
    print("Each touch is labeled by the physical target, but ALL five piezos are recorded.")
    print("When ARMED appears, perform exactly one instructed gesture.\n")
    source = SerialTouchSource(args.port, args.baud, HARDWARE_NUM_SENSORS, HARDWARE_NUM_SAMPLES)
    saved = 0
    try:
        for gesture in args.gestures:
            print(f"\n=== {gesture.upper()}: {GESTURE_INSTRUCTIONS[gesture]} ===")
            for placement in body_map.placements:
                target = placement.name.replace("_", " ")
                print(f"\nTarget: Sensor {placement.channel + 1} / GPIO {placement.pin} — {target}")
                for repetition in range(1, args.count + 1):
                    input(f"[{repetition}/{args.count}] Press Enter when your hand is away from the robot...")
                    source.reset_input_buffer()
                    waveform = source.get_triggered_touch(
                        args.threshold,
                        baseline_samples=150,
                        pretrigger_samples=60,
                        on_armed=lambda: print("  ARMED — touch now!", flush=True),
                    )
                    path = save_touch(
                        waveform,
                        gesture,
                        placement.name,
                        source="real",
                        sample_rate=HARDWARE_SAMPLE_RATE,
                    )
                    saved += 1
                    print(f"  Saved {path.name} ({saved}/{total}). Remove your hand.")
    except KeyboardInterrupt:
        print(f"\nStopped safely. {saved} examples were kept.")
    finally:
        source.close()

    print("\nCollection complete.")
    print("Train with: python -m ml.train --source real --sensor-count 5")


if __name__ == "__main__":
    main()
