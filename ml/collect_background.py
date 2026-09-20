"""Record the powered-on robot with nobody touching it."""

from __future__ import annotations

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
    parser.add_argument("--port", required=True, help="ESP32 serial port, for example COM5")
    parser.add_argument("--count", type=int, default=100, help="Number of 0.30-second windows")
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD)
    args = parser.parse_args()
    if args.count < 20:
        parser.error("Collect at least 20 windows; 100 is recommended.")

    print("POWERED-ON BACKGROUND COLLECTION")
    print("Leave the robot on in its normal operating state.")
    print("Do not touch the robot, table, cable, or sensors while this runs.")
    print("Existing recordings will not be overwritten; new rows are appended.")
    input("Press Enter when the robot is untouched and ready...")

    source = SerialTouchSource(args.port, args.baud, HARDWARE_NUM_SENSORS, HARDWARE_NUM_SAMPLES)
    saved = 0
    try:
        source.reset_input_buffer()
        for index in range(args.count):
            waveform = source.get_touch()
            save_touch(
                waveform,
                touch_type="no_touch",
                location="none",
                source="real",
                sample_rate=HARDWARE_SAMPLE_RATE,
            )
            saved += 1
            if saved == 1 or saved % 10 == 0 or saved == args.count:
                print(f"Saved quiet window {saved}/{args.count}")
    except KeyboardInterrupt:
        print(f"\nStopped safely. {saved} background windows were kept.")
    finally:
        source.close()

    print("\nBackground collection complete.")
    print("Retrain with: python -m ml.train --source real --sensor-count 5")


if __name__ == "__main__":
    main()
