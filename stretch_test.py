"""Make a Unitree Go2 do its front-leg stretch once and disconnect."""

from __future__ import annotations

import argparse
import ipaddress
import os
import subprocess
import time

from dimos.robot.unitree.connection import UnitreeWebRTCConnection


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
        raise SystemExit(
            "The AES key must be exactly 32 hexadecimal characters."
        )

    return aes_key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make the Go2 do its front-leg stretch once"
    )

    parser.add_argument("ip", nargs="?", help="Go2 IP address")
    parser.add_argument(
        "--robot-ip",
        help="Alternative way to provide the Go2 IP",
    )

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

    print(f"Go2 answered at {robot_ip}. Connecting...")

    robot = None

    try:
        robot = UnitreeWebRTCConnection(
            ip=robot_ip,
            aes_128_key=aes_key,
        )

        time.sleep(2)

        confirmation = input(
            "Place the Go2 on a clear, level floor and keep its controller ready.\n"
            "Type STRETCH to perform the front-leg stretch once: "
        )

        if confirmation.strip() != "STRETCH":
            print("Cancelled. No action was requested.")
            return

        # Make sure the robot starts from a normal standing position.
        print("Standing normally...")
        robot.sport_command(1004)  # StandUp
        time.sleep(3)

        # Built-in Unitree dog-stretch / bow motion.
        print("Stretching front legs...")
        robot.sport_command(1017)  # Stretch

        # Give the built-in animation enough time to complete.
        time.sleep(6)

        print("Stretch complete.")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        if robot is not None:
            robot.disconnect()

        print("Go2 connection closed.")


if __name__ == "__main__":
    main()
    