"""Display a live flattened Go2 torso map and estimated touch position."""

import argparse
from collections import deque

import numpy as np

from .body_map import BodySurfaceMap, SENSOR_MAP_PATH
from .config import (
    HARDWARE_NUM_SAMPLES,
    HARDWARE_NUM_SENSORS,
    SERIAL_BAUD,
)
from .source_serial import SerialTouchSource


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD)
    parser.add_argument("--history", type=int, default=300)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    body_map = BodySurfaceMap.load()
    if not SENSOR_MAP_PATH.exists():
        print("WARNING: using provisional GPIO-to-position ordering.")
        print("Run python -m ml.calibrate_map for an accurate physical map.")
    source = SerialTouchSource(
        args.port, args.baud, HARDWARE_NUM_SENSORS, HARDWARE_NUM_SAMPLES
    )
    history: deque[np.ndarray] = deque(maxlen=args.history)

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.set_xlim(0, 1)
    axis.set_ylim(1, 0)
    axis.set_xlabel("Body length: head/front → tail/rear")
    axis.set_ylabel("Wrapped surface: top → right → underside → left → top")
    axis.set_title("Skinless: flattened Go2 torso touch map")
    axis.set_xticks(np.linspace(0, 1, 11), minor=True)
    axis.set_yticks(np.linspace(0, 1, 9), minor=True)
    axis.grid(which="both", alpha=0.25)
    axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["top", "right", "underside", "left", "top"])

    for placement in body_map.placements:
        axis.scatter(placement.u, placement.v, s=90, color="tab:blue")
        axis.annotate(
            f"{placement.name}\nGPIO {placement.pin}",
            (placement.u, placement.v),
            xytext=(5, 5),
            textcoords="offset points",
        )
    touch_marker = axis.scatter([], [], s=240, color="tab:red", marker="x", linewidths=3)
    status = axis.text(0.01, 1.04, "Waiting for sensor data...", transform=axis.transAxes)
    figure.tight_layout()
    plt.ion()
    plt.show(block=False)
    plt.pause(0.1)

    try:
        counter = 0
        while plt.fignum_exists(figure.number):
            history.append(source.read_sample(max_wait_seconds=0.5))
            counter += 1
            if counter % 10 or len(history) < 30:
                continue
            values = np.stack(history)
            baseline = np.median(values[: max(10, len(values) // 3)], axis=0)
            strengths = np.max(np.abs(values[-150:] - baseline), axis=0)
            if float(np.max(strengths)) < args.threshold:
                status.set_text("Idle — touch the torso")
                figure.canvas.flush_events()
                continue
            result = body_map.localize_strengths(strengths)
            u, v = float(result["u"]), float(result["v"])
            touch_marker.set_offsets([[u, v]])
            region = body_map.region_name(u, v)
            status.set_text(
                f"Estimated touch: {region} | u={u:.2f}, v={v:.2f} | "
                f"confidence={float(result['confidence']):.0%}"
            )
            figure.canvas.draw_idle()
            figure.canvas.flush_events()
            plt.pause(0.001)
    except KeyboardInterrupt:
        print("\nLive map stopped.")
    finally:
        source.close()


if __name__ == "__main__":
    main()
