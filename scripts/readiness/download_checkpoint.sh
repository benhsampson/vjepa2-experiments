#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec "${BASE_PYTHON:-/usr/local/bin/python}" scripts/readiness/download_checkpoint.py
