"""Associate GPIO channels with the five marked Go2 body positions."""

import argparse

import numpy as np
from scipy.optimize import linear_sum_assignment

from .body_map import (
    BodySurfaceMap,
    CALIBRATION_TARGETS,
    FIXED_CHANNEL_TARGETS,
    SENSOR_MAP_PATH,
    SensorPlacement,
)
from .config import (
    HARDWARE_NUM_SAMPLES,
    HARDWARE_NUM_SENSORS,
    HARDWARE_SENSOR_PINS,
    SERIAL_BAUD,
)
from .preprocessing import subtract_baseline
from .source_serial import SerialTouchSource


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD)
    args = parser.parse_args()

    source = SerialTouchSource(
        args.port, args.baud, HARDWARE_NUM_SENSORS, HARDWARE_NUM_SAMPLES
    )
    captured_peaks: list[np.ndarray] = []
    print("This calibrates all five physical sensor positions.")
    print("Sensor 5 / GPIO 15 is fixed at rear_top_butt (the top butt panel).")
    print("Keep the body still between touches. Press Ctrl+C to cancel.\n")
    try:
        for target in CALIBRATION_TARGETS:
            friendly_name = str(target["name"]).replace("_", " ")
            input(f"Press Enter, then make ONE clear tap at {friendly_name}...")
            waveform = source.get_triggered_touch(args.threshold, 100, 50)
            corrected, _ = subtract_baseline(waveform)
            peaks = np.max(np.abs(corrected), axis=0)
            captured_peaks.append(peaks)
            strongest = int(np.argmax(peaks))
            print(
                f"Captured peaks: {peaks.tolist()} "
                f"(strongest this tap: GPIO {HARDWARE_SENSOR_PINS[strongest]})"
            )
    except KeyboardInterrupt:
        print("\nCalibration cancelled; no map was saved.")
        return
    finally:
        source.close()

    # A touch can excite multiple nearby piezos. Solve all five assignments
    # together instead of greedily reusing whichever channel happened to be
    # strongest for one individual tap.
    score_matrix = np.stack(captured_peaks)
    target_names = [str(target["name"]) for target in CALIBRATION_TARGETS]
    fixed_pairs = [
        (target_names.index(target_name), channel)
        for channel, target_name in FIXED_CHANNEL_TARGETS.items()
    ]
    fixed_rows = {row for row, _ in fixed_pairs}
    fixed_channels = {channel for _, channel in fixed_pairs}
    remaining_rows = [row for row in range(len(CALIBRATION_TARGETS)) if row not in fixed_rows]
    remaining_channels = [channel for channel in range(HARDWARE_NUM_SENSORS) if channel not in fixed_channels]
    sub_rows, sub_channels = linear_sum_assignment(
        -score_matrix[np.ix_(remaining_rows, remaining_channels)]
    )
    inferred_pairs = [
        (remaining_rows[int(row)], remaining_channels[int(channel)])
        for row, channel in zip(sub_rows, sub_channels, strict=True)
    ]
    assignments = fixed_pairs + inferred_pairs
    placements: list[SensorPlacement] = []
    print("\nBest one-to-one sensor assignment:")
    for target_row, channel in assignments:
        target = CALIBRATION_TARGETS[int(target_row)]
        pin = HARDWARE_SENSOR_PINS[int(channel)]
        score = float(score_matrix[target_row, channel])
        fixed_note = " [fixed hardware position]" if channel in FIXED_CHANNEL_TARGETS else ""
        print(f"  {target['name']}: Sensor {channel + 1} / GPIO {pin} (peak {score:.1f}){fixed_note}")
        placements.append(
            SensorPlacement(
                channel=int(channel),
                pin=pin,
                name=str(target["name"]),
                u=float(target["u"]),
                v=float(target["v"]),
            )
        )

    BodySurfaceMap(placements).save(SENSOR_MAP_PATH)
    print(f"\nSaved calibrated map to {SENSOR_MAP_PATH}")


if __name__ == "__main__":
    main()
