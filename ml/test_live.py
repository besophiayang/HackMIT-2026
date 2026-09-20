"""Verify live ESP32 sensors with a scrolling signal/location display."""

import argparse
from collections import deque
import time

import numpy as np

from .config import (
    HARDWARE_NUM_SAMPLES,
    HARDWARE_NUM_SENSORS,
    HARDWARE_SENSOR_PINS,
    SERIAL_BAUD,
)
from .source_serial import SerialTouchSource


def run_text_display(source: SerialTouchSource) -> None:
    print("Reading sensors. Press Ctrl+C to stop.")
    last_print = 0.0
    while True:
        sample = source.read_sample(max_wait_seconds=5)
        now = time.monotonic()
        if now - last_print >= 0.05:
            values = "  ".join(
                f"S{i + 1}/GPIO{pin}: {sample[i]:7.1f}"
                for i, pin in enumerate(HARDWARE_SENSOR_PINS)
            )
            print(values, end="\r")
            last_print = now


def run_plot(
    source: SerialTouchSource, history_length: int, strength_window: int
) -> None:
    import matplotlib.pyplot as plt

    history: deque[np.ndarray] = deque(maxlen=history_length)
    figure, (signal_axis, strength_axis) = plt.subplots(2, 1, figsize=(11, 8))
    labels = [
        f"Sensor {index + 1} (GPIO {pin})"
        for index, pin in enumerate(HARDWARE_SENSOR_PINS)
    ]
    lines = []
    for label in labels:
        line, = signal_axis.plot([], [], label=label, linewidth=1.2)
        lines.append(line)
    bars = strength_axis.bar(labels, np.zeros(HARDWARE_NUM_SENSORS))

    signal_axis.set_title("Skinless: live piezo signals")
    signal_axis.set_xlabel("Recent samples")
    signal_axis.set_ylabel("Raw ADC reading")
    signal_axis.legend(loc="upper left", ncols=2)
    signal_axis.grid(alpha=0.25)
    strength_axis.set_title("Current vibration strength (baseline-corrected)")
    strength_axis.set_ylabel("Absolute ADC deviation")
    strength_axis.tick_params(axis="x", rotation=20)
    strength_axis.grid(axis="y", alpha=0.25)
    status_text = figure.text(
        0.5,
        0.01,
        "Opening serial stream...",
        ha="center",
        color="tab:orange",
    )
    figure.tight_layout()
    plt.ion()
    plt.show(block=False)
    # Let Tk create the window before waiting for the first serial sample.
    plt.pause(0.1)
    print("Touch near each sensor. Close the graph or press Ctrl+C to stop.")

    sample_counter = 0
    while plt.fignum_exists(figure.number):
        try:
            history.append(source.read_sample(max_wait_seconds=0.25))
            status_text.set_text("Receiving five-sensor data")
            status_text.set_color("tab:green")
        except TimeoutError as exc:
            # Keep the graph open so connection/firmware problems are visible.
            status_text.set_text(str(exc))
            status_text.set_color("tab:red")
            figure.canvas.draw_idle()
            figure.canvas.flush_events()
            plt.pause(0.05)
            continue
        sample_counter += 1
        if sample_counter % 10 != 0 or len(history) < 10:
            continue

        values = np.stack(history)
        x = np.arange(len(values))
        baseline = np.median(values[: max(1, len(values) // 3)], axis=0)
        # Piezo impulses may last only a few samples. Display the largest
        # deviation in a recent window so a touch is not missed between redraws.
        recent = values[-min(strength_window, len(values)) :]
        deviation = np.max(np.abs(recent - baseline), axis=0)
        strongest = int(np.argmax(deviation))
        status_text.set_text(
            f"Receiving five-sensor data — strongest: {labels[strongest]} "
            f"(peak {deviation[strongest]:.0f})"
        )

        for sensor, line in enumerate(lines):
            line.set_data(x, values[:, sensor])
        for sensor, bar in enumerate(bars):
            bar.set_height(float(deviation[sensor]))
            bar.set_color(
                "tab:red"
                if sensor == strongest and deviation[sensor] > 0
                else "tab:blue"
            )

        signal_axis.relim()
        signal_axis.autoscale_view()
        strength_axis.set_ylim(0, max(10.0, float(np.max(deviation)) * 1.25))
        figure.canvas.draw_idle()
        figure.canvas.flush_events()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="ESP32 COM port, for example COM5")
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD)
    parser.add_argument("--plot", action="store_true", help="Show the five-sensor live graph")
    parser.add_argument("--history", type=int, default=300, help="Samples visible in the graph")
    parser.add_argument(
        "--strength-window",
        type=int,
        default=150,
        help="Recent samples used for touch-strength bars",
    )
    args = parser.parse_args()
    if args.history < 10:
        parser.error("--history must be at least 10")
    if args.strength_window < 1:
        parser.error("--strength-window must be at least 1")

    try:
        source = SerialTouchSource(
            args.port, args.baud, HARDWARE_NUM_SENSORS, HARDWARE_NUM_SAMPLES
        )
    except ConnectionError as exc:
        print(f"ERROR: {exc}")
        print("Close Arduino Serial Monitor/Plotter and any older viewer, then try again.")
        return
    try:
        if args.plot:
            run_plot(source, args.history, args.strength_window)
        else:
            run_text_display(source)
    except KeyboardInterrupt:
        print("\nStopped.")
    except (ConnectionError, TimeoutError) as exc:
        print(f"\nERROR: {exc}")
    finally:
        source.close()


if __name__ == "__main__":
    main()
