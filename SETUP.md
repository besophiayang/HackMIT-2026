# Unitree Go2 command starter (Windows + WSL2 + dimOS)

The development environment is installed and verified:

- WSL2 with Ubuntu 24.04
- Python 3.12
- dimOS in `/home/betiff/dimos-app`
- dimOS Go2 blueprints and Unitree support

The file to edit is `robot_program.py`. It uses the Go2's high-level controller;
it does not control individual motors.

## Connect and run from VS Code

1. Connect the Go2 and Windows computer to the same non-guest Wi-Fi.
2. Obtain the Go2 IP address from the Unitree app or router.
3. Open a new VS Code terminal. The **Go2 Ubuntu** profile automatically opens
   Ubuntu and activates `/home/betiff/dimos-app/.venv`.
4. Either run `python robot_program.py --robot-ip <address>`, or select
   **Terminal > Run Task**, then **Go2: connect and run starter**.
5. Enter the robot's IP address when requested.

The program first pings the robot, starts dimOS, and prints its available skills.
It will not accept movement commands until `READY` is typed exactly.

You can also run it from PowerShell:

```powershell
.\run_robot.ps1 -RobotIp 192.168.1.123
```

Replace the example address with the real Go2 address.

## Safety

- Put the robot on a clear floor, away from people, pets, stairs, and roads.
- Keep the physical controller ready.
- Test `stop` first.
- Movement is limited to two seconds per command and starts at 0.2 m/s.
- Voice recognition should be added only after typed commands work reliably.
