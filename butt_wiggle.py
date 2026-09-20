"""Make a Unitree Go2 perform a short in-place happy butt wiggle."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import subprocess
import time

from dimos.robot.unitree.connection import UnitreeWebRTCConnection
from unitree_webrtc_connect.constants import RTC_TOPIC

EULER_API_ID = 1007
WIGGLE_REPETITIONS = 5
WIGGLE_YAW_RADIANS = 0.18
WIGGLE_ROLL_RADIANS = 0.04
HALF_WIGGLE_SECONDS = 0.28


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


def set_body_pose(
    robot: UnitreeWebRTCConnection,
    *,
    roll: float,
    yaw: float,
) -> bool:
    response = robot.publish_request(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": EULER_API_ID,
            "parameter": json.dumps({"x": roll, "y": 0.0, "z": yaw}),
        },
    )
    return bool(response)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make the Go2 perform a short in-place butt wiggle"
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
            "Place the Go2 on a clear, level, non-slip floor and keep its "
            "controller ready.\nType WIGGLE to perform the in-place happy "
            "butt wiggle: "
        )
        if confirmation.strip().upper() != "WIGGLE":
            print("Cancelled. No action was requested.")
            return

        print("Entering balanced standing...")
        robot.balance_stand()
        time.sleep(1)

        print("Wiggling in place...")
        for _ in range(WIGGLE_REPETITIONS):
            if not set_body_pose(
                robot,
                roll=WIGGLE_ROLL_RADIANS,
                yaw=WIGGLE_YAW_RADIANS,
            ):
                print("The Go2 rejected the posture command.")
                return
            time.sleep(HALF_WIGGLE_SECONDS)

            if not set_body_pose(
                robot,
                roll=-WIGGLE_ROLL_RADIANS,
                yaw=-WIGGLE_YAW_RADIANS,
            ):
                print("The Go2 rejected the posture command.")
                return
            time.sleep(HALF_WIGGLE_SECONDS)

        set_body_pose(robot, roll=0.0, yaw=0.0)
        time.sleep(0.5)
        robot.balance_stand()
        time.sleep(1)
        print("Butt wiggle complete.")
    except KeyboardInterrupt:
        print("\nInterrupted; returning to balanced standing.")
    finally:
        if robot is not None:
            set_body_pose(robot, roll=0.0, yaw=0.0)
            robot.balance_stand()
            time.sleep(0.5)
            robot.disconnect()
        print("Go2 connection closed.")


if __name__ == "__main__":
    main()
