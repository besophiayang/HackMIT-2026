"""Approach a detected hand and stop when it fills enough of the camera view.

This is a standalone controller. It does not import or modify the emote files,
and only one robot-control program should be run at a time.
"""

from __future__ import annotations

import argparse
import ctypes
import ipaddress
import math
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.robot.unitree.connection import UnitreeWebRTCConnection

DEFAULT_MODEL = Path("/home/betiff/dimos-app/hand_landmarker.task")
MEDIAPIPE_LIBRARY_DIR = Path(
    "/home/betiff/dimos-app/.mediapipe-libs/usr/lib/x86_64-linux-gnu"
)
GESTURE_HOLD_SECONDS = 0.0
HAND_CLOSE_AREA_RATIO = 0.18
MAX_TRAVEL_METERS = 2.0
CAMERA_STALE_SECONDS = 1.0
HAND_CONFIDENCE = 0.10
APPROACH_SPEED_MPS = 0.30
MAX_TURN_RADIANS_PER_SECOND = 0.12
STEERING_GAIN = 0.40
CONTROL_PERIOD_SECONDS = 0.10
MOVEMENT_PERIOD_SECONDS = 0.05
COMMAND_DEBUG_PERIOD_SECONDS = 0.50
HAND_LOST_GRACE_SECONDS = 2.0
CENTER_DEADZONE = 0.10
CAMERA_VIEW_PORT = 8765


class LatestSensors:
    """Thread-safe storage for the newest camera and odometry data."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.image: Any | None = None
        self.image_received = 0.0
        self.pose: Any | None = None

    def on_image(self, image: Any) -> None:
        with self._lock:
            self.image = image
            self.image_received = time.monotonic()

    def on_pose(self, pose: Any) -> None:
        with self._lock:
            self.pose = pose

    def snapshot(self) -> tuple[Any, float, Any]:
        with self._lock:
            return (
                self.image,
                self.image_received,
                self.pose,
            )


class DesiredControl:
    """Thread-safe velocity requested by the vision controller."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.desired_vx = 0.0
        self.desired_yaw = 0.0
        self.pause_reason: str | None = None
        self.stop_reason: str | None = None

    def set_motion(self, vx: float, yaw: float) -> None:
        with self._lock:
            if self.stop_reason is None:
                self.desired_vx = vx
                self.desired_yaw = yaw
                self.pause_reason = None

    def snapshot(self) -> tuple[float, float]:
        with self._lock:
            return self.desired_vx, self.desired_yaw

    def stop(self, reason: str) -> bool:
        """Set the desired velocity to zero and report the first stop reason."""
        with self._lock:
            self.desired_vx = 0.0
            self.desired_yaw = 0.0
            if self.stop_reason is not None:
                return False
            self.stop_reason = reason
        print(f"STOP REASON: {reason}")
        return True

    def pause(self, reason: str) -> None:
        """Stop walking without ending the command thread or connection."""
        with self._lock:
            changed = (
                self.desired_vx != 0.0
                or self.desired_yaw != 0.0
                or self.pause_reason != reason
            )
            self.desired_vx = 0.0
            self.desired_yaw = 0.0
            self.pause_reason = reason
        if changed:
            print(f"PAUSE REASON: {reason}")


class BrowserCameraView:
    """Serve the latest annotated frame as a local MJPEG browser page."""

    def __init__(self, port: int = CAMERA_VIEW_PORT) -> None:
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self.stop_requested = threading.Event()
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/":
                    page = (
                        b"<!doctype html><html><head><title>Go2 Camera</title>"
                        b"<style>body{background:#111;color:#eee;font-family:sans-serif;"
                        b"text-align:center}img{max-width:95vw;max-height:80vh}"
                        b"button{font-size:1.2rem;padding:.7rem 1.4rem;margin:1rem}</style>"
                        b"</head><body><h2>Go2 front camera</h2>"
                        b"<img src='/stream.mjpg'><br>"
                        b"<button onclick=\"fetch('/stop',{method:'POST'})\">"
                        b"Stop robot program</button></body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)
                    return
                if self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                    )
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        while not viewer.stop_requested.is_set():
                            jpeg = viewer.latest_jpeg()
                            if jpeg is None:
                                time.sleep(0.05)
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                            self.wfile.write(
                                f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                            )
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                            time.sleep(0.05)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self.send_error(404)

            def do_POST(self) -> None:
                if self.path != "/stop":
                    self.send_error(404)
                    return
                viewer.stop_requested.set()
                self.send_response(204)
                self.end_headers()

            def log_message(self, _format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def update(self, frame: np.ndarray[Any, Any]) -> None:
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if success:
            with self._lock:
                self._latest_jpeg = encoded.tobytes()

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def close(self) -> None:
        self.stop_requested.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1.0)


def open_camera_page() -> None:
    url = f"http://localhost:{CAMERA_VIEW_PORT}"
    print(f"Live camera view: {url}")
    try:
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-Command", "Start-Process", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        print("Open the camera URL above in a Windows browser.")


def require_reachable_robot(robot_ip: str) -> None:
    try:
        ipaddress.ip_address(robot_ip)
    except ValueError as exc:
        raise SystemExit(f"Invalid robot IP address: {robot_ip}") from exc

    result = subprocess.run(
        ["ping", "-c", "1", "-W", "2", robot_ip],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"The Go2 did not answer at {robot_ip}.")


def require_aes_key(aes_key: str | None) -> str:
    if not aes_key:
        raise SystemExit("Set UNITREE_AES_128_KEY before running this file.")
    try:
        key_bytes = bytes.fromhex(aes_key)
    except ValueError as exc:
        raise SystemExit(
            "The AES key must be exactly 32 hexadecimal characters."
        ) from exc
    if len(key_bytes) != 16:
        raise SystemExit("The AES key must be exactly 32 hexadecimal characters.")
    return aes_key


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def create_hand_detector(model_path: Path) -> Any:
    existing_path = os.environ.get("LD_LIBRARY_PATH", "")
    library_path = str(MEDIAPIPE_LIBRARY_DIR)
    os.environ["LD_LIBRARY_PATH"] = (
        f"{library_path}:{existing_path}" if existing_path else library_path
    )
    ctypes.CDLL(
        str(MEDIAPIPE_LIBRARY_DIR / "libGLESv2.so.2"),
        mode=ctypes.RTLD_GLOBAL,
    )

    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        num_hands=2,
        min_hand_detection_confidence=HAND_CONFIDENCE,
        min_hand_presence_confidence=HAND_CONFIDENCE,
        min_tracking_confidence=HAND_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hand(
    detector: Any,
    rgb_image: np.ndarray[Any, Any],
) -> tuple[bool, float | None, float | None, np.ndarray[Any, Any]]:
    """Detect actual hand landmarks and return center and image-area ratio."""
    import mediapipe as mp

    contiguous_rgb = np.ascontiguousarray(rgb_image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=contiguous_rgb)
    result = detector.detect(mp_image)
    annotated = cv2.cvtColor(contiguous_rgb, cv2.COLOR_RGB2BGR)
    if not result.hand_landmarks:
        return False, None, None, annotated

    height, width = annotated.shape[:2]
    connections = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
    )
    best_center: float | None = None
    best_area_ratio = 0.0

    for hand in result.hand_landmarks:
        xs = np.array([landmark.x for landmark in hand], dtype=np.float32)
        ys = np.array([landmark.y for landmark in hand], dtype=np.float32)
        x_min, x_max = float(np.min(xs)), float(np.max(xs))
        y_min, y_max = float(np.min(ys)), float(np.max(ys))
        area_ratio = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
        if area_ratio > best_area_ratio:
            best_area_ratio = area_ratio
            best_center = (x_min + x_max) / 2.0

        pixels = [
            (int(landmark.x * width), int(landmark.y * height)) for landmark in hand
        ]
        for start, end in connections:
            cv2.line(annotated, pixels[start], pixels[end], (50, 220, 50), 2)
        for point in pixels:
            cv2.circle(annotated, point, 3, (0, 255, 255), -1)
        cv2.rectangle(
            annotated,
            (int(x_min * width), int(y_min * height)),
            (int(x_max * width), int(y_max * height)),
            (255, 120, 0),
            2,
        )

    return True, best_center, best_area_ratio, annotated


def odometry_distance(start_pose: Any, current_pose: Any) -> float:
    return math.hypot(
        float(current_pose.x) - float(start_pose.x),
        float(current_pose.y) - float(start_pose.y),
    )


def stop_robot(
    robot: UnitreeWebRTCConnection,
    control: DesiredControl,
    movement_stop: threading.Event,
    reason: str,
) -> None:
    """Issue one terminal stop and prevent further movement commands."""
    first_stop = control.stop(reason)
    movement_stop.set()
    if first_stop:
        robot.stop_movement()


def movement_loop(
    robot: UnitreeWebRTCConnection,
    control: DesiredControl,
    movement_stop: threading.Event,
) -> None:
    """Continuously transmit the most recent vision-requested velocity at 20 Hz."""
    last_debug = 0.0
    while not movement_stop.is_set():
        loop_started = time.monotonic()
        vx, yaw = control.snapshot()
        command = Twist(
            linear=(vx, 0.0, 0.0),
            angular=(0.0, 0.0, yaw),
        )
        if not robot.move(command):
            stop_robot(robot, control, movement_stop, "movement command rejected")
            return
        if loop_started - last_debug >= COMMAND_DEBUG_PERIOD_SECONDS:
            print(f"CMD | vx={vx:.2f} | yaw={yaw:.2f}")
            last_debug = loop_started
        elapsed = time.monotonic() - loop_started
        movement_stop.wait(max(0.0, MOVEMENT_PERIOD_SECONDS - elapsed))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Approach a sustained low-hand gesture using the front camera"
    )
    parser.add_argument("ip", nargs="?", help="Go2 IP address")
    parser.add_argument("--robot-ip", help="Alternative way to provide the Go2 IP")
    parser.add_argument(
        "--aes-key",
        default=os.environ.get("UNITREE_AES_128_KEY"),
        help="Prefer setting UNITREE_AES_128_KEY instead",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to an Ultralytics human-pose model",
    )
    parser.add_argument(
        "--observe-only",
        action="store_true",
        help="Detect the gesture and show the camera without moving",
    )
    parser.add_argument(
        "--approach-once",
        action="store_true",
        help="Arm one approach immediately, with no second confirmation prompt",
    )
    args = parser.parse_args()

    if args.observe_only and args.approach_once:
        parser.error("Use either --observe-only or --approach-once, not both")

    robot_ip = args.robot_ip or args.ip or os.environ.get("ROBOT_IP")
    if not robot_ip:
        parser.error("Provide the Go2 IP address or set ROBOT_IP")
    if not args.model.is_file():
        raise SystemExit(f"Hand model not found: {args.model}")

    require_reachable_robot(robot_ip)
    aes_key = require_aes_key(args.aes_key)
    print("Loading the dedicated MediaPipe hand model...")
    detector = create_hand_detector(args.model)

    robot = None
    camera_view: BrowserCameraView | None = None
    subscriptions: list[Any] = []
    sensors = LatestSensors()
    control = DesiredControl()
    movement_stop = threading.Event()
    movement_thread: threading.Thread | None = None
    try:
        print(f"Go2 answered at {robot_ip}. Connecting without moving...")
        robot = UnitreeWebRTCConnection(
            ip=robot_ip,
            aes_128_key=aes_key,
            velocity_api=True,
        )
        subscriptions.extend(
            (
                robot.video_stream().subscribe(sensors.on_image),
                robot.odom_stream().subscribe(sensors.on_pose),
            )
        )

        if args.approach_once:
            print("One approach is armed. Waiting to see a low outstretched hand...")
        elif args.observe_only:
            confirmation_word = "OBSERVE"
            prompt = "Type OBSERVE to begin detection without robot movement: "
        else:
            confirmation_word = "APPROACH"
            prompt = (
                "Clear at least 2 metres around the Go2 and person, keep the "
                "controller ready, and stand directly ahead.\nType APPROACH to "
                "arm autonomous slow movement: "
            )
        if (
            not args.approach_once
            and input(prompt).strip().upper() != confirmation_word
        ):
            print("Cancelled. No autonomous behavior was armed.")
            return

        print("Waiting for camera and odometry streams...")
        stream_deadline = time.monotonic() + 15.0
        while time.monotonic() < stream_deadline:
            image, _, pose = sensors.snapshot()
            if image is not None and pose is not None:
                break
            time.sleep(0.1)
        else:
            print("Required sensor streams were not all received. No movement occurred.")
            return

        camera_view = BrowserCameraView()
        open_camera_page()
        if not args.observe_only:
            if not robot.free_walk():
                raise RuntimeError("The Go2 rejected FreeWalk mode")
            print("FreeWalk velocity mode ready. Waiting for a hand...")
        print("Ready. Hold one hand low and toward the camera.")
        gesture_started: float | None = None
        approaching = False
        start_pose: Any | None = None
        last_hand_seen = 0.0
        last_hand_x = 0.5
        last_hand_area_ratio = 0.0
        last_status = 0.0
        last_tracking_status = 0.0

        while True:
            loop_started = time.monotonic()
            image, image_time, pose = sensors.snapshot()

            if image is None or loop_started - image_time > CAMERA_STALE_SECONDS:
                stop_robot(robot, control, movement_stop, "camera stream stale")
                break
            if pose is None:
                stop_robot(robot, control, movement_stop, "odometry unavailable")
                break
            rgb = image.to_rgb().as_numpy()
            hand_detected, hand_x, hand_area_ratio, annotated = detect_hand(detector, rgb)
            status = "HAND DETECTED" if hand_detected else "Waiting for hand"
            cv2.putText(
                annotated,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if hand_detected else (0, 200, 255),
                2,
            )
            camera_view.update(annotated)
            if camera_view.stop_requested.is_set():
                stop_robot(robot, control, movement_stop, "operator stop button")
                break
            if hand_detected and hand_x is not None:
                last_hand_seen = loop_started
                last_hand_x = hand_x
                if hand_area_ratio is not None:
                    last_hand_area_ratio = hand_area_ratio

            if not approaching:
                if hand_detected:
                    if gesture_started is None:
                        gesture_started = loop_started
                    held_for = loop_started - gesture_started
                    if loop_started - last_status > 0.8:
                        visible_area = 0.0 if hand_area_ratio is None else hand_area_ratio
                        print(
                            f"Low hand detected | hand_area={visible_area:.3f} | "
                            f"close_threshold={HAND_CLOSE_AREA_RATIO:.3f}"
                        )
                        last_status = loop_started
                    if held_for >= GESTURE_HOLD_SECONDS:
                        if args.observe_only:
                            print("Gesture confirmed. Observe-only test successful; no movement sent.")
                            break
                        if (
                            hand_area_ratio is not None
                            and hand_area_ratio >= HAND_CLOSE_AREA_RATIO
                        ):
                            control.pause("hand already close; waiting for a distant hand")
                            gesture_started = None
                            continue
                        approaching = True
                        start_pose = pose
                        control.set_motion(APPROACH_SPEED_MPS, 0.0)
                        if movement_thread is None or not movement_thread.is_alive():
                            movement_thread = threading.Thread(
                                target=movement_loop,
                                args=(robot, control, movement_stop),
                                daemon=True,
                                name="go2-movement",
                            )
                            movement_thread.start()
                        print("Beginning slow camera-guided approach...")
                else:
                    gesture_started = None
            else:
                hand_age = loop_started - last_hand_seen
                if hand_age > HAND_LOST_GRACE_SECONDS:
                    control.pause("hand lost; continuing to search")
                    approaching = False
                    gesture_started = None
                    print("Waiting for another hand without closing the Go2 connection.")
                    continue
                elif last_hand_area_ratio >= HAND_CLOSE_AREA_RATIO:
                    print(
                        f"hand_area={last_hand_area_ratio:.3f} reached "
                        f"close_threshold={HAND_CLOSE_AREA_RATIO:.3f}"
                    )
                    control.pause("hand close; target reached")
                    approaching = False
                    gesture_started = None
                    print("Target reached. Continuing to look for another distant hand.")
                    continue
                elif start_pose is not None and odometry_distance(start_pose, pose) > MAX_TRAVEL_METERS:
                    stop_robot(robot, control, movement_stop, "maximum travel distance")
                    break

                horizontal_error = last_hand_x - 0.5
                if abs(horizontal_error) < CENTER_DEADZONE:
                    turn_speed = 0.0
                else:
                    turn_speed = clamp(
                        -horizontal_error * STEERING_GAIN,
                        -MAX_TURN_RADIANS_PER_SECOND,
                        MAX_TURN_RADIANS_PER_SECOND,
                    )
                control.set_motion(APPROACH_SPEED_MPS, turn_speed)
                if loop_started - last_tracking_status >= 0.5:
                    current_hand_x = "None" if hand_x is None else f"{hand_x:.2f}"
                    print(
                        f"Tracking | hand_x={current_hand_x} | "
                        f"last_hand_x={last_hand_x:.2f} | "
                        f"hand_area={last_hand_area_ratio:.3f} | "
                        f"vx={APPROACH_SPEED_MPS:.2f} | yaw={turn_speed:.2f} | "
                        f"hand_age={hand_age:.2f}s"
                    )
                    last_tracking_status = loop_started
                if movement_stop.is_set():
                    break

            elapsed = time.monotonic() - loop_started
            time.sleep(max(0.0, CONTROL_PERIOD_SECONDS - elapsed))
    except KeyboardInterrupt:
        if robot is not None:
            stop_robot(robot, control, movement_stop, "KeyboardInterrupt")
    finally:
        if robot is not None:
            stop_robot(robot, control, movement_stop, "program shutdown/error")
        if movement_thread is not None:
            movement_thread.join(timeout=1.0)
        for subscription in subscriptions:
            subscription.dispose()
        if camera_view is not None:
            camera_view.close()
        detector.close()
        if robot is not None:
            robot.disconnect()
        print("Go2 connection closed.")


if __name__ == "__main__":
    main()
