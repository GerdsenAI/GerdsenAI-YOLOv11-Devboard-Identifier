#!/bin/bash
# Demo launcher for Teensy 4.1 Binary Classifier
# Activates environment, finds best model, runs camera inference

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use system Python (has CUDA support on Jetson)
# Note: venv on Jetson has CPU-only PyTorch, system Python has CUDA
PYTHON="python3"

# Find best model (prefer binary classifier)
MODEL_PATH=""

# Check consolidated binary classifier models first
if [ -d "runs/detect/runs/binary_teensy" ]; then
    MODEL_PATH=$(find runs/detect/runs/binary_teensy -name "best.pt" -type f 2>/dev/null | head -1)
fi

# Check teensy_41_pipeline models
if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    if [ -d "teensy_41_pipeline/runs" ]; then
        MODEL_PATH=$(find teensy_41_pipeline/runs -name "best.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    fi
fi

# Fallback to main pipeline models
if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    if [ -d "runs" ]; then
        MODEL_PATH=$(find runs -name "best.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    fi
fi

# Check models directory
if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    if [ -f "models/devboard_best.pt" ]; then
        MODEL_PATH="models/devboard_best.pt"
    fi
fi

if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: No trained model found."
    echo "Train a model first:"
    echo "  cd teensy_41_pipeline && python3 05_train_teensy.py"
    exit 1
fi

echo "============================================"
echo "Dev Board Identifier Demo"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Press 'q' to quit, 'r' to toggle recording"
echo "============================================"

# Run inference using the main scripts inference (updated with path fixes)
$PYTHON scripts/06_run_inference.py \
    --model "$MODEL_PATH" \
    --camera 0 \
    --confidence 0.5
