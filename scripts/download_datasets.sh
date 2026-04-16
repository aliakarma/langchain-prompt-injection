#!/usr/bin/env bash
# scripts/download_datasets.sh
# Downloads external datasets if user wants full benchmark.
# Gracefully skips if unavailable.

set -e

echo "=====================================-"
echo "  Downloading Evaluation Datasets     "
echo "=====================================-"

DATA_DIR="data"
EXTERNAL_RAW="$DATA_DIR/external_raw"
mkdir -p "$EXTERNAL_RAW"

echo "Attempting to download HackAPrompt dataset (optional)..."
if curl -fsSL -o "$EXTERNAL_RAW/hackaprompt_raw.jsonl" "https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset/resolve/main/hackaprompt.jsonl"; then
    echo "[OK] HackAPrompt dataset downloaded successfully."
else
    echo "[WARN] Failed to download HackAPrompt. Skipping..."
fi

echo "Attempting to download BIPIA dataset (optional)..."
if curl -fsSL -o "$EXTERNAL_RAW/bipia_raw.jsonl" "https://raw.githubusercontent.com/Example/bipia/main/dataset.jsonl"; then
    echo "[OK] BIPIA dataset downloaded successfully."
else
    echo "[WARN] Failed to download BIPIA dataset. Skipping..."
fi

echo "Attempting to download JailbreakBench dataset (optional)..."
if curl -fsSL -o "$EXTERNAL_RAW/jailbreakbench_raw.jsonl" "https://raw.githubusercontent.com/JailbreakBench/JailbreakBench/main/data/jailbreakbench.jsonl"; then
    echo "[OK] JailbreakBench dataset downloaded successfully."
else
    echo "[WARN] Failed to download JailbreakBench dataset. Skipping..."
fi

echo ""
echo "Note: If you are running offline, the repository will automatically fallback"
echo "to the small, fully-included sample datasets inside 'data/sample/'."
echo "You can now run 'make benchmark-full' to use these downloaded datasets,"
echo "or 'make benchmark' for the minimal, offline-ready benchmarking."
