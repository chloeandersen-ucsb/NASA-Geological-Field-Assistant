#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Resolve CAPSTONE_ROOT placeholder in config for convenience
CAP_ROOT="${CAPSTONE_ROOT:-$ROOT/..}"
sed -i "s|<CAPSTONE_ROOT>|$CAP_ROOT|g" capstone_connector/config.yaml

export CAPSTONE_CONNECTOR_CONFIG="$ROOT/capstone_connector/config.yaml"
python3 -m capstone_connector.server
