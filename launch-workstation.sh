#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
umask 077
ROOT="${SC500K_HOME:-$HOME/.local/share/servitaror-sc100k}"
mkdir -p "$ROOT"
ROOT="$(realpath "$ROOT")"
command -v python3 >/dev/null || { echo "Install Python 3.10+ and python3-venv"; exit 1; }
command -v cc >/dev/null || { echo "Install build-essential (gcc and binutils)"; exit 1; }
if [ ! -x "$ROOT/venv/bin/python" ]; then python3 -m venv "$ROOT/venv"; fi
PY="$ROOT/venv/bin/python"
if ! "$PY" -c 'import numpy; assert int(numpy.__version__.split(".")[0]) == 2' >/dev/null 2>&1; then
  "$PY" -m pip install -r requirements.txt
fi
# Optional --image PATH selects a user-supplied Linux image; otherwise use
# the exact historical RV32IMA image. Download only on initial setup.
IMAGE="$ROOT/Image"
if [[ "${1:-}" == "--image" ]]; then
  IMAGE="$(realpath "$2")"; shift 2
else
  HASH=9feee94f9fbb3f9a761485fb982c78977594d9c0db8b0b411b5677dfb5efc325
  if [[ ! -f "$IMAGE" ]]; then
    command -v curl >/dev/null || { echo "Install curl or supply --image PATH"; exit 1; }
    tmp="$(mktemp "$ROOT/Image.XXXXXX")"
    trap 'rm -f "$tmp"' EXIT
    curl -fL --retry 2 https://raw.githubusercontent.com/bane9/RISCVBox/main/linux/prebuilt/Image -o "$tmp"
    echo "$HASH  $tmp" | sha256sum -c -
    mv "$tmp" "$IMAGE"
    trap - EXIT
  fi
  echo "$HASH  $IMAGE" | sha256sum -c -
fi
SESSION="$(mktemp -d "$ROOT/session.XXXXXXXX")"
echo "Session: $SESSION"
echo "Checkpoint: $SESSION/checkpoint.npz"
export OPENBLAS_NUM_THREADS=1
exec "$PY" neural_workstation_r40.py --image "$IMAGE" --build-dir "$SESSION/native" --checkpoint "$SESSION/checkpoint.npz" "$@"

