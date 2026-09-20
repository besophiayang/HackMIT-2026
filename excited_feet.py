"""Make a Unitree Go2 perform a brief happy-feet stepping reaction."""

from __future__ import annotations

import argparse
import ipaddress
import os
import subprocess
import time

from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.robot.unitree.connection import UnitreeWebRTCConnection

STATIC_WALK_API_ID = 1061
STEP_REPETITIONS = 2
STEP_SPEED_METERS_PER_SECOND = 0.12
HALF_STEP_SECONDS = 0.45


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make the Go2 perform a short happy-feet stepping reaction"
    )
    parser.add_argument("ip", nargs="?", help="Go2 IP address")
    parser.add_argument("--robot-ip", help="Alternative way to provide the Go2 IP")
    parser.add_argument(
        "--aes-key",
        default=os.environ.get("UNITREE_AES_128_KEY"),
        help="Prefer setting UNITREE_AES_128_KEY instead",
    )
    args = parser.parse_args()

    robot_ip = args.robot_ip or args.ip or os.environ.get("ROBOT_IP")
    if not robot_ip:
        parser.error("Provide the Go2 IP address or set ROBOT_IP")

    require_reachable_robot(robot_ip)
    aes_key = require_aes_key(args.aes_key)

    print(f"Go2 answered at {robot_ip}. Connecting without moving...")
    robot = None
    try:
        robot = UnitreeWebRTCConnection(ip=robot_ip, aes_128_key=aes_key)
        time.sleep(2)

        confirmation = input(
            "Place the Go2 on a clear, level, non-slip floor with at least "
            "half a metre clear in front and behind it. Keep its controller "
            "ready.\nType EXCITED to perform two tiny forward-back step pairs: "
        )
        if confirmation.strip().upper() != "EXCITED":
            print("Cancelled. No action was requested.")
            return

        robot.balance_stand()
        time.sleep(1)
        print("Selecting Unitree's slow StaticWalk gait...")
        if not robot.sport_command(STATIC_WALK_API_ID):
            print("The Go2 rejected StaticWalk.")
            return
        time.sleep(0.5)

        print("Performing the happy-feet steps...")

        tiny_forward = Twist(linear=(STEP_SPEED_METERS_PER_SECOND, 0.0, 0.0))
        tiny_backward = Twist(linear=(-STEP_SPEED_METERS_PER_SECOND, 0.0, 0.0))

        for _ in range(STEP_REPETITIONS):
            if not robot.move(tiny_forward, duration=HALF_STEP_SECONDS):
                print("The Go2 rejected the forward step command.")
                return
            if not robot.move(tiny_backward, duration=HALF_STEP_SECONDS):
                print("The Go2 rejected the backward step command.")
                return

        robot.stop_movement()
        time.sleep(0.5)
        robot.balance_stand()
        time.sleep(1)
        print("Excited-feet reaction complete.")
    except KeyboardInterrupt:
        print("\nInterrupted; stopping movement.")
    finally:
        if robot is not None:
            robot.stop_movement()
            robot.balance_stand()
            time.sleep(0.5)
            robot.disconnect()
        print("Go2 connection closed.")


if __name__ == "__main__":
    main()
