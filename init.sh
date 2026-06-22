#!/usr/bin/env bash
set -e

# Install uv if not already present
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Creating virtual environment and installing dependencies..."
uv sync

# torch is pulled from the CUDA 12.6 index (see pyproject.toml) so it runs on the
# course's Tesla M60 GPUs. Verify CUDA is actually usable — otherwise NER training
# silently falls back to CPU and is far slower. A non-fatal heads-up, not an error.
echo ""
echo "Checking GPU / CUDA availability..."
uv run python - <<'PY' || true
import torch
if torch.cuda.is_available():
    print(f"  OK: CUDA available -> {torch.cuda.get_device_name(0)} (torch {torch.__version__})")
else:
    try:
        build_archs = torch._C._cuda_getArchFlags()  # compile-time, populated even w/o a GPU
    except Exception:
        build_archs = "?"
    print(f"  WARNING: torch {torch.__version__} does not see a GPU; NER will train on CPU (slow).")
    print(f"           this torch build supports: {build_archs}")
    print("           Check `nvidia-smi`, and that this build covers the GPU's compute capability.")
PY

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Scripts live in code/. Run them from this directory, e.g.:"
echo "  uv run python code/train_tokenizer.py --domain_file data/domain_1_train.txt --output_dir tokenizers"
echo "generate_tokenizers.py and check_submission.py are at the root (no code/ prefix):"
echo "  uv run python generate_tokenizers.py"
