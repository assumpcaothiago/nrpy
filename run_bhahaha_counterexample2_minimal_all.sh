#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

PYTHON_BIN="${PYTHON:-python}"
XDG_CACHE_HOME=/tmp HOME=/tmp MPLCONFIGDIR=/tmp \
  "$PYTHON_BIN" nrpy/examples/tests/bhahaha_counterexample2_minimal_run_all.py "$@"
