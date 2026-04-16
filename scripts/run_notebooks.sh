#!/usr/bin/env bash
# scripts/run_notebooks.sh
# Tests reproducibility of all Jupyter notebooks.

set -e

echo "=====================================-"
echo "  Running Jupyter Notebooks cleanly   "
echo "=====================================-"

if ! command -v jupyter >/dev/line 2>&1; then
    echo "jupyter not found. Ensure you have installed dev dependencies."
    exit 1
fi

NOTEBOOK_DIR="notebooks"

for nb in $(ls $NOTEBOOK_DIR/*.ipynb | sort); do
    echo "[TEST] Running $nb..."
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 "$nb"
    echo "[OK] $nb completed successfully."
done

echo ""
echo "All notebooks executed completely and are updated in-place."
