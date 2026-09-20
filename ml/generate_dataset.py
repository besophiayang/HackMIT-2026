"""Command-line synthetic dataset generator."""

import argparse
import shutil

import pandas as pd

from .config import LABELS_PATH, LOCATIONS, RAW_DATA_DIR, SAMPLE_RATE, TOUCH_TYPES
from .source_simulated import SimulatedTouchSource
from .storage import save_touch


def generate_dataset(examples_per_class: int = 30, overwrite: bool = False) -> None:
    if examples_per_class < 1:
        raise ValueError("examples-per-class must be at least 1.")

    existing_files = list(RAW_DATA_DIR.glob("*.npy")) if RAW_DATA_DIR.exists() else []
    if LABELS_PATH.exists() or existing_files:
        if not overwrite:
            raise FileExistsError(
                "Dataset already exists. Use --overwrite only to replace a purely "
                "simulated dataset."
            )
        if LABELS_PATH.exists():
            labels = pd.read_csv(LABELS_PATH)
            if "source" not in labels or (labels["source"] != "simulated").any():
                raise RuntimeError("Refusing to overwrite a dataset containing real data.")
        if RAW_DATA_DIR.exists():
            shutil.rmtree(RAW_DATA_DIR)
        LABELS_PATH.unlink(missing_ok=True)

    source = SimulatedTouchSource()
    total = len(TOUCH_TYPES) * len(LOCATIONS) * examples_per_class
    completed = 0
    for touch_type in TOUCH_TYPES:
        for location in LOCATIONS:
            for _ in range(examples_per_class):
                waveform = source.get_touch(touch_type, location)
                save_touch(waveform, touch_type, location, source="simulated", sample_rate=SAMPLE_RATE)
                completed += 1

    print(f"Generated {completed} synthetic examples (expected {total}).")
    print(f"Waveforms: {RAW_DATA_DIR}")
    print(f"Labels:    {LABELS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-per-class", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate_dataset(args.examples_per_class, args.overwrite)


if __name__ == "__main__":
    main()
