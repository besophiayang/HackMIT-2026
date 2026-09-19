#!/usr/bin/env bash
set -euo pipefail

echo "Installing dimOS library environment for Unitree Go2..."
curl -fsSL https://raw.githubusercontent.com/dimensionalOS/dimos/main/scripts/install.sh | bash -s -- \
  --mode library \
  --project-dir "$HOME/dimos-app" \
  --non-interactive \
  --no-nix \
  --no-cuda \
  --no-sysctl \
  --skip-tests \
  --extras base,unitree

source "$HOME/dimos-app/.venv/bin/activate"
uv run dimos --help

echo
echo "dimOS installation completed and verified."
echo "You can close this window and return to Codex."
exec bash
