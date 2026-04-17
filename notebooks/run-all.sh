#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BASELINES_DIR="$SCRIPT_DIR/baselines"
RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="$BASELINES_DIR/out/$RUN_TIMESTAMP"
VENV_DIR="$REPO_DIR/.venv"
KERNEL_NAME="python3"

if [ -x "$VENV_DIR/bin/python" ]; then
  PAPERMILL_CMD=("$VENV_DIR/bin/python" -m papermill)
else
  PAPERMILL_CMD=("papermill")
fi

NOTEBOOKS=(
  "if_model.ipynb"
  "pca_model.ipynb"
  "gmm_model.ipynb"
  "ocsvm_model.ipynb"
  "rf_model.ipynb"
  "xgb_model.ipynb"
)

if ! command -v "${PAPERMILL_CMD[0]}" >/dev/null 2>&1; then
  echo "papermill is not installed or not available on PATH." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

cd "$BASELINES_DIR"

echo "Running baseline notebooks with papermill."
echo "Output notebooks will be written to: $OUTPUT_DIR"
echo "Using Jupyter kernel: $KERNEL_NAME"

for notebook in "${NOTEBOOKS[@]}"; do
  output_path="$OUTPUT_DIR/$notebook"

  echo
  echo "=== Executing $notebook ==="
  "${PAPERMILL_CMD[@]}" -k "$KERNEL_NAME" "$notebook" "$output_path"
done

echo
echo "Finished running all baseline notebooks."
