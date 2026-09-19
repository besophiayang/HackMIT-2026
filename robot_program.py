"""Safe starter program for controlling a Unitree Go2 through dimOS.

Run this file inside the installed Ubuntu/WSL dimOS environment. Starting the
program connects to the robot but does not move it. A movement happens only
after a command is typed and confirmed.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import subprocess
import time

from dimos import Dimos


def require_reachable_robot(robot_ip: str) -> None:
    """Stop early if the address is invalid or the robot cannot be reached."""
    try:
        ipaddress.ip_address(robot_ip)
    except ValueError as exc:
        raise SystemExit(f"Invalid robot IP address: {robot_ip}") from exc

    result = subprocess.run(
        ["ping", "-c", "1", "-W", "2", robot_ip],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"The Go2 did not answer at {robot_ip}. Confirm that the robot and "
            "computer are on the same non-guest Wi-Fi network."
        )


def run_command(app: Dimos, command: str) -> bool:
    """Run one small, bounded movement. Return False to exit."""
    command = command.lower().strip()

    if command in {"quit", "exit", "stop"}:
        return False
    if command in {"forward", "walk", "walk forward"}:
        app.skills.relative_move(forward=0.20)
    elif command in {"back", "backward", "walk backward"}:
        app.skills.relative_move(forward=-0.20)
    elif command in {"left", "turn left"}:
        app.skills.relative_move(degrees=15.0)
    elif command in {"right", "turn right"}:
        app.skills.relative_move(degrees=-15.0)
    else:
        print("Unknown command. Try: forward, backward, left, right, stop")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Unitree Go2 dimOS starter")
    parser.add_argument(
        "--robot-ip",
        default=os.environ.get("ROBOT_IP"),
        help="Go2 IP address (or set ROBOT_IP)",
    )
    args = parser.parse_args()

    if not args.robot_ip:
        parser.error("Provide --robot-ip or set the ROBOT_IP environment variable")

    require_reachable_robot(args.robot_ip)
    os.environ["ROBOT_IP"] = args.robot_ip

    print(f"Go2 answered at {args.robot_ip}.")
    print("Starting the dimOS Go2 connection. The robot will not move yet...")

    app = Dimos(n_workers=8)
    try:
        app.run("unitree-go2-agentic")
        time.sleep(8)
        print("dimOS started. Available skills:")
        print(app.skills)

        confirmation = input(
            "\nClear the floor and keep the controller/app ready. "
            "Type READY to enable typed movement commands: "
        )
        if confirmation.strip() != "READY":
            print("Not enabled; disconnecting safely.")
            return

        print("Commands: forward, backward, left, right, stop")
        while run_command(app, input("go2> ")):
            pass
    except KeyboardInterrupt:
        print("\nInterrupted; stopping...")
    finally:
        app.stop()
        print("dimOS stopped.")


if __name__ == "__main__":
    main()
