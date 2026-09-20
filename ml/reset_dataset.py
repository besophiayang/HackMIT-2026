"""Safely remove a selected subset of recordings while preserving other data."""

from __future__ import annotations

import argparse
from datetime import datetime
import shutil

import pandas as pd

from .config import LABELS_PATH, RAW_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["real", "simulated"], required=True)
    parser.add_argument("--sensor-count", type=int, required=True)
    parser.add_argument("--confirm", action="store_true", help="Actually delete; otherwise preview only")
    args = parser.parse_args()

    if not LABELS_PATH.exists():
        raise FileNotFoundError("data/labels.csv does not exist.")
    labels = pd.read_csv(LABELS_PATH)
    if "num_sensors" not in labels.columns:
        raise ValueError("labels.csv has no num_sensors column.")
    selected = (labels["source"] == args.source) & (labels["num_sensors"] == args.sensor_count)
    rows = labels[selected]
    print(f"Matched {len(rows)} {args.source} recordings with {args.sensor_count} sensors.")
    if rows.empty:
        return
    print("Files:", ", ".join(rows["file"].head(5)), "..." if len(rows) > 5 else "")
    if not args.confirm:
        print("Preview only. Add --confirm to remove exactly these recordings.")
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = LABELS_PATH.with_name(f"labels.backup-{stamp}.csv")
    archive = LABELS_PATH.parent / f"archived-{args.source}-{args.sensor_count}sensors-{stamp}"
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(LABELS_PATH, backup)
    for filename in rows["file"]:
        path = RAW_DATA_DIR / str(filename)
        if path.parent.resolve() != RAW_DATA_DIR.resolve():
            raise ValueError(f"Unsafe dataset path in labels.csv: {filename}")
        if path.exists():
            shutil.move(str(path), archive / path.name)
    labels[~selected].to_csv(LABELS_PATH, index=False)
    print(f"Removed {len(rows)} recordings from the active dataset.")
    print(f"Waveforms were moved to recoverable archive: {archive.name}")
    print(f"Labels backup: {backup.name}")


if __name__ == "__main__":
    main()
