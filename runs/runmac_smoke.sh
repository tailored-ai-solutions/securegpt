#!/bin/bash
set -euo pipefail

# Fast Apple Silicon sanity check. This does not download datasets or train.

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    if [ -x "$HOME/.local/bin/uv" ]; then
        UV="$HOME/.local/bin/uv"
    else
        echo "uv is not installed. Run:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
else
    UV="$(command -v uv)"
fi

"$UV" sync --extra cpu --group dev

"$UV" run python - <<'PY'
import torch
from nanochat.common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, autodetect_device_type

print("torch:", torch.__version__)
print("mps built:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())
print("nanochat device:", autodetect_device_type())
print("nanochat dtype:", COMPUTE_DTYPE, f"({COMPUTE_DTYPE_REASON})")
PY

"$UV" run pytest
