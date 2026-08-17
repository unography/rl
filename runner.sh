#!/usr/bin/env bash

set -Eeuo pipefail

TORCHRL_DIR="${1:-$HOME/rl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/nightly/cu130}"

cd "$TORCHRL_DIR"

command -v uv >/dev/null 2>&1 || {
    echo "Error: uv is not installed or not available on PATH." >&2
    exit 1
}

test -f pyproject.toml || {
    echo "Error: $TORCHRL_DIR does not appear to be the TorchRL repository." >&2
    exit 1
}

echo "Creating Python $PYTHON_VERSION environment..."
uv python install "$PYTHON_VERSION"
uv venv --clear --python "$PYTHON_VERSION" .venv

PYTHON="$TORCHRL_DIR/.venv/bin/python"

echo "Installing CUDA 13 nightly PyTorch..."
uv pip install \
    --python "$PYTHON" \
    --prerelease allow \
    --index-url "$PYTORCH_INDEX" \
    torch torchvision

echo "Installing build dependencies..."
uv pip install \
    --python "$PYTHON" \
    pip setuptools wheel \
    "pybind11[global]" cmake ninja numpy packaging

echo "Installing TensorDict from GitHub..."
uv pip install \
    --python "$PYTHON" \
    "git+https://github.com/pytorch/tensordict.git"

echo "Installing TorchRL in editable mode..."
uv pip install \
    --python "$PYTHON" \
    --no-build-isolation \
    --editable .

echo "Installing contributor dependencies..."
uv pip install \
    --python "$PYTHON" \
    --group dev

echo "Installing test and DeepMind Control dependencies..."
uv pip install \
    --python "$PYTHON" \
    --editable ".[dm_control,tests]"

echo "Installing pre-commit hooks..."
"$TORCHRL_DIR/.venv/bin/pre-commit" install

echo "Checking dependencies..."
"$PYTHON" -m pip check

echo "Checking TorchRL and CUDA..."
"$PYTHON" - <<'PY'
import torch
import torchrl
import tensordict

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("TorchRL:", torchrl.__file__)
print("TensorDict:", tensordict.__version__)

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access CUDA")

print("GPU:", torch.cuda.get_device_name(0))

tensor = torch.randn(1024, 1024, device="cuda")
print("CUDA tensor test:", tensor.sum().item())
PY

echo
echo "TorchRL contributor environment is ready."
echo "Activate it with:"
echo "  source $TORCHRL_DIR/.venv/bin/activate"
