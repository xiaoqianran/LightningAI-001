#!/usr/bin/env bash
# Local S1 smoke on CPU (or GPU if available). Uses sample data by default.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="${PYTHON:-python3}"
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export MOE_ARTIFACT_DIR="${MOE_ARTIFACT_DIR:-${ROOT}/artifacts}"

MAX_STEPS="${MAX_STEPS:-100}"
FORCE_SAMPLE="${FORCE_SAMPLE:-1}"

echo "=== S1 local smoke ==="
echo "python=$PY max_steps=$MAX_STEPS artifacts=$MOE_ARTIFACT_DIR"

if [[ "$FORCE_SAMPLE" == "1" ]]; then
  "$PY" -m moe_zh.prepare_d0 --force-sample --max-bytes 500000
else
  "$PY" -m moe_zh.prepare_d0 --max-bytes "${MAX_BYTES:-5000000}"
fi

"$PY" -m moe_zh.train --resume none --max-steps "$MAX_STEPS"
"$PY" -m moe_zh.generate

echo "=== artifacts ==="
ls -la "$MOE_ARTIFACT_DIR" || true
ls -la "$MOE_ARTIFACT_DIR/checkpoints" || true
echo "OK"
