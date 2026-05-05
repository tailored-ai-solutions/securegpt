# SecureGPT Mac Setup

This fork tracks `karpathy/nanochat` and is set up for local experimentation on Apple Silicon Macs.

## Repo Remotes

- `origin`: `https://github.com/tailored-ai-solutions/securegpt.git`
- `upstream`: `https://github.com/karpathy/nanochat.git`

To pull future upstream changes:

```bash
git fetch upstream
git merge upstream/master
git push origin master
```

## Install

The repo uses `uv` and requires Python 3.10+. On a fresh Mac:

```bash
cd /Users/alexgo/code/securegpt
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra cpu --group dev
```

The `cpu` extra is also the right choice for MPS on Mac. PyTorch will use Apple Silicon MPS when nanochat autodetects it.

## Verify

```bash
uv run python -c "import torch; from nanochat.common import autodetect_device_type; print(torch.__version__); print(torch.backends.mps.is_available()); print(autodetect_device_type())"
uv run pytest
```

Expected on the M5/M3 machines:

- `torch.backends.mps.is_available()` prints `True`.
- `autodetect_device_type()` prints `mps`.
- The test suite passes, with CUDA-only tests skipped.

## Mac Training Entry Point

Use the upstream Mac/CPU example:

```bash
WANDB_RUN=dummy bash runs/runcpu.sh
```

That script is intentionally tiny compared with the full 8xH100 speedrun. It is good for local learning and code-path testing, not for training a strong model.

For a lighter check before downloading data or training:

```bash
bash runs/runmac_smoke.sh
```
