"""Central configuration for data capture, training, and inference."""

from pathlib import Path

USE_HARDWARE = False

SERIAL_PORT = "COM5"
SERIAL_BAUD = 1_000_000

# Current ESP32-S3 hardware configuration. Keep these separate from the
# simulator settings below: real data is never padded to eight sensors.
HARDWARE_SENSOR_PINS = [4, 5, 6, 7, 15]
HARDWARE_NUM_SENSORS = len(HARDWARE_SENSOR_PINS)
HARDWARE_SAMPLE_RATE = 1_000
HARDWARE_WINDOW_SECONDS = 0.30
HARDWARE_NUM_SAMPLES = int(HARDWARE_SAMPLE_RATE * HARDWARE_WINDOW_SECONDS)

NUM_SENSORS = 8
SAMPLE_RATE = 2_000
WINDOW_SECONDS = 0.30
NUM_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)

TOUCH_TYPES = ["tap", "hard_tap", "poke", "scratch", "stroke", "pet"]
TOUCH_CLASS_GROUPS = {
    "tap": "tap_poke",
    "poke": "tap_poke",
    "hard_tap": "hard_tap",
    "scratch": "scratch",
    "stroke": "pat_stroke",
    "pet": "pat_stroke",
}
LOCATIONS = ["head", "front_left", "front_right", "mid_back", "rear"]

# Paths are based on this file, so commands work from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
LABELS_PATH = DATA_DIR / "labels.csv"
MODELS_DIR = PROJECT_ROOT / "models"
