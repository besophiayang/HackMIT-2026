"""Collect real left/right directional strokes for continuous localization."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone

import numpy as np

from .config import DATA_DIR, HARDWARE_NUM_SENSORS, HARDWARE_SAMPLE_RATE, SERIAL_BAUD
from .source_serial import SerialTouchSource


PATHS = [
    ("left", "front_to_back"),
    ("left", "back_to_front"),
    ("right", "front_to_back"),
    ("right", "back_to_front"),
]
PATH_DIR = DATA_DIR / "motion_paths"
LABELS_PATH = DATA_DIR / "motion_labels.csv"
FIELDS = ["file", "side", "direction", "sample_rate", "num_sensors", "timestamp"]


def _next_path() -> tuple[str, object]:
    PATH_DIR.mkdir(parents=True, exist_ok=True)
    used = [int(path.stem.split("_")[-1]) for path in PATH_DIR.glob("motion_*.npy")]
    index = max(used, default=-1) + 1
    name = f"motion_{index:05d}.npy"
    return name, PATH_DIR / name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--count", type=int, default=20, help="Examples per side/direction")
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD)
    args = parser.parse_args()
    if args.count < 10:
        parser.error("Use at least 10 examples per path; 20 is recommended.")

    sample_count = int(HARDWARE_SAMPLE_RATE * 0.60)
    try:
        source = SerialTouchSource(args.port, args.baud, HARDWARE_NUM_SENSORS, sample_count)
    except ConnectionError as error:
        raise SystemExit(
            f"Could not use {args.port}. Stop the Skinless dashboard and close Arduino "
            f"Serial Monitor/Plotter, then run this command again.\nDetails: {error}"
        ) from None
    total = len(PATHS) * args.count
    saved = 0
    print(f"Will collect {total} directional strokes. Existing data is preserved.")
    print("Use one finger, stay on the requested side, and traverse the torso in about 0.5 seconds.")
    try:
        for side, direction in PATHS:
            start, end = ("front", "rear") if direction == "front_to_back" else ("rear", "front")
            print(f"\n=== {side.upper()} SIDE: {start.upper()} -> {end.upper()} ===")
            for repetition in range(1, args.count + 1):
                input(f"[{repetition}/{args.count}] Hand away; press Enter to arm...")
                source.reset_input_buffer()
                waveform = source.get_triggered_touch(
                    args.threshold,
                    baseline_samples=150,
                    pretrigger_samples=25,
                    on_armed=lambda: print("  ARMED — make one smooth stroke now!", flush=True),
                )
                name, path = _next_path()
                np.save(path, waveform, allow_pickle=False)
                needs_header = not LABELS_PATH.exists() or LABELS_PATH.stat().st_size == 0
                with LABELS_PATH.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=FIELDS)
                    if needs_header:
                        writer.writeheader()
                    writer.writerow({
                        "file": name,
                        "side": side,
                        "direction": direction,
                        "sample_rate": HARDWARE_SAMPLE_RATE,
                        "num_sensors": HARDWARE_NUM_SENSORS,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                saved += 1
                print(f"  Saved {name} ({saved}/{total})")
    except KeyboardInterrupt:
        print(f"\nStopped safely. {saved} paths were kept.")
    finally:
        source.close()
    print("\nTrain the spatial model with: python -m ml.train_motion")


if __name__ == "__main__":
    main()
