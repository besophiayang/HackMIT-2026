"""Interactively make a Unitree Go2 sit, rise, or disconnect."""

from __future__ import annotations

import argparse
import ipaddress
import os
import subprocess
import time

from dimos.robot.unitree.connection import UnitreeWebRTCConnection

SIT_API_ID = 1009
RISE_SIT_API_ID = 1010


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
    parser = argparse.ArgumentParser(description="Control the Go2 sitting position")
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

        print("Place the Go2 on a clear, level, non-slip floor.")
        print("Commands: SIT = sit down, UP = rise, QUIT = disconnect")

        while True:
            command = input("Command (SIT/UP/QUIT): ").strip().upper()

            if command == "SIT":
                print("Sitting...")
                if robot.sport_command(SIT_API_ID):
                    print("Sit command accepted. The Go2 will remain sitting.")
                else:
                    print("The Go2 rejected the Sit command.")
                time.sleep(3)
            elif command == "UP":
                print("Getting up...")
                if robot.sport_command(RISE_SIT_API_ID):
                    print("Rise command accepted.")
                else:
                    print("The Go2 rejected the RiseSit command.")
                time.sleep(4)
            elif command == "QUIT":
                print("Disconnecting. The robot will remain in its current posture.")
                break
            else:
                print("Unknown command. Type SIT, UP, or QUIT.")
    except KeyboardInterrupt:
        print("\nInterrupted. Disconnecting without changing posture.")
    finally:
        if robot is not None:
            robot.disconnect()
        print("Go2 connection closed.")


if __name__ == "__main__":
    main()
