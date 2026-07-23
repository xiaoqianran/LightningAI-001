#!/usr/bin/env bash
# 01 — Basic Lightning AI batch job via CLI (hello world).
set -euo pipefail

export PATH="${HOME}/.local/bin:/config/.local/bin:${PATH}"

OWNER="${LIGHTNING_ORG:-${LIGHTNING_USER:-${LIGHTNING_USERNAME:-seachenxyt}}}"
TEAMSPACE="${LIGHTNING_TEAMSPACE:-seachenxyt}"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
NAME="${LIGHTNING_JOB_NAME:-hello-cpu-${STAMP}}"

# Personal teamspace: pass --org only when LIGHTNING_ORG is set.
ORG_ARGS=()
if [[ -n "${LIGHTNING_ORG:-}" ]]; then
  ORG_ARGS=(--org "${LIGHTNING_ORG}")
else
  # CLI accepts --user for user-owned teamspaces
  ORG_ARGS=(--user "${OWNER}")
fi

echo "=== LightningAI-001 / 01 hello job (CLI) ==="
echo "teamspace : ${OWNER}/${TEAMSPACE}"
echo "job name  : ${NAME}"
echo

if ! command -v lightning >/dev/null 2>&1; then
  echo "error: 'lightning' CLI not found. Install: pip install -U lightning-sdk" >&2
  exit 1
fi

if [[ -z "${LIGHTNING_USER_ID:-}" || -z "${LIGHTNING_API_KEY:-}" ]]; then
  if [[ ! -f "${HOME}/.lightning/credentials.json" ]]; then
    echo "warning: no LIGHTNING_* env and no ~/.lightning/credentials.json — try 'lightning login'" >&2
  fi
fi

lightning job run \
  --name "${NAME}" \
  --teamspace "${TEAMSPACE}" \
  "${ORG_ARGS[@]}" \
  --image "python:3.11-slim" \
  --machine CPU \
  --command 'python -c "print(\"Hello from Lightning Job!\"); import platform,sys; print(sys.version.split()[0], platform.platform())"'

echo
echo "submitted: ${NAME}"
echo "list     : lightning job list --teamspace ${OWNER}/${TEAMSPACE}"
echo "inspect  : lightning job inspect ${NAME} --teamspace ${OWNER}/${TEAMSPACE}"
echo
echo "Tip: SDK path waits + prints logs: python examples/01_hello_job.py"
