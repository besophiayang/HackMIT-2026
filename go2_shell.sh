#!/usr/bin/env bash
set -e

DIMOS_ENV="$HOME/dimos-app/.venv/bin/activate"
PROJECT_DIR="/mnt/c/Users/betif/OneDrive/Desktop/Skinless"

if [[ ! -f "$DIMOS_ENV" ]]; then
    echo "dimOS environment not found at $DIMOS_ENV" >&2
    exit 1
fi

source "$DIMOS_ENV"
cd "$PROJECT_DIR"

echo "Go2 development shell ready."
echo "Python: $(python --version)"
echo "dimOS:  $(dimos --version 2>/dev/null || command -v dimos)"
echo
echo "Next, after the Go2 joins your Wi-Fi:"
echo "  dimos go2tool discover"
echo "  export ROBOT_IP=<robot-ip>"
echo "  ping \"\$ROBOT_IP\""
echo "  python robot_program.py --robot-ip \"\$ROBOT_IP\""
echo

exec bash -i
