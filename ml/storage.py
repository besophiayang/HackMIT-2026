"""Store simulated and real touches in one interchangeable format."""

import csv
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import LABELS_PATH, RAW_DATA_DIR, SAMPLE_RATE
from .source_base import TouchSource


LABEL_FIELDS = ["file", "touch_type", "location", "source", "sample_rate", "num_sensors", "timestamp"]


def _next_filename(raw_dir: Path, source: str) -> str:
    pattern = re.compile(r"touch_(\d+)\.npy$")
    used = []
    for path in raw_dir.glob("touch_*.npy"):
        match = pattern.match(path.name)
        if match:
            used.append(int(match.group(1)))
    prefix = "real" if source == "real" else "touch"
    next_index = max(used, default=-1) + 1
    while (raw_dir / f"{prefix}_{next_index:05d}.npy").exists():
        next_index += 1
    return f"{prefix}_{next_index:05d}.npy"


def _ensure_label_schema(labels_path: Path) -> None:
    """Upgrade the old four-column CSV without dropping existing rows."""
    if not labels_path.exists() or labels_path.stat().st_size == 0:
        return
    with labels_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == LABEL_FIELDS:
            return
        rows = list(reader)
    temporary = labels_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for row in rows:
            row.setdefault("sample_rate", str(SAMPLE_RATE) if row.get("source") == "simulated" else "")
            row.setdefault("num_sensors", "8" if row.get("source") == "simulated" else "")
            row.setdefault("timestamp", "")
            writer.writerow({field: row.get(field, "") for field in LABEL_FIELDS})
    temporary.replace(labels_path)


def save_touch(
    waveform: np.ndarray,
    touch_type: str,
    location: str,
    source: str = "real",
    sample_rate: int = SAMPLE_RATE,
    raw_dir: Path = RAW_DATA_DIR,
    labels_path: Path = LABELS_PATH,
) -> Path:
    """Save a validated waveform and atomically choose its unique name."""
    TouchSource.validate_waveform(waveform)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    raw_dir.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    _ensure_label_schema(labels_path)
    filename = _next_filename(raw_dir, source)
    output_path = raw_dir / filename
    np.save(output_path, waveform, allow_pickle=False)

    needs_header = not labels_path.exists() or labels_path.stat().st_size == 0
    try:
        with labels_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=LABEL_FIELDS
            )
            if needs_header:
                writer.writeheader()
            writer.writerow(
                {
                    "file": filename,
                    "touch_type": touch_type,
                    "location": location,
                    "source": source,
                    "sample_rate": sample_rate,
                    "num_sensors": waveform.shape[1],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    return output_path
