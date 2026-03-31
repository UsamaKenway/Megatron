#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all.sh  —  Full BF16 vs INT4 vs NVFP4 benchmark pipeline
#
# Usage:
#   cd /mnt/c/MachineLearning/UsamaKenway/nvfp4_/our_solution
#   bash run_all.sh
#
# Output files produced in this directory:
#   ckpt_bf16.pt         BF16 training checkpoint
#   ckpt_nvfp4.pt        NVFP4 training checkpoint
#   results.json         Raw metrics (all phases)
#   benchmark_report.md  Final comparison table + analysis
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANOGPT_DIR="$SCRIPT_DIR/../nanoGPT"
DATA_DIR="$NANOGPT_DIR/data/shakespeare_char"

echo "============================================================"
echo " NanoGPT Precision Benchmark"
echo " BF16  →  INT4 PTQ  →  NVFP4 Native"
echo "============================================================"
echo ""

# ── Step 0: Prepare dataset ───────────────────────────────────────────────────
if [ ! -f "$DATA_DIR/train.bin" ]; then
    echo "[0/4] Downloading & tokenising shakespeare_char dataset…"
    python "$DATA_DIR/prepare.py"
else
    echo "[0/4] Dataset already prepared — skipping."
fi
echo ""

# ── Step 1: BF16 baseline ─────────────────────────────────────────────────────
echo "[1/4] Phase 1: BF16 training  (1 000 iters)…"
python "$SCRIPT_DIR/train_bf16.py"
echo ""

# ── Step 2: INT4 PTQ ──────────────────────────────────────────────────────────
echo "[2/4] Phase 2: INT4 Post-Training Quantisation…"
python "$SCRIPT_DIR/quantize_int4.py"
echo ""

# ── Step 3: NVFP4 native training ─────────────────────────────────────────────
echo "[3/4] Phase 3: NVFP4 native training  (1 000 iters)…"
python "$SCRIPT_DIR/train_nvfp4.py"
echo ""

# ── Step 4: Generate report ───────────────────────────────────────────────────
echo "[4/4] Phase 4: Generating benchmark report…"
python "$SCRIPT_DIR/generate_report.py"
echo ""

echo "============================================================"
echo " All phases complete!"
echo ""
echo " Results  : $SCRIPT_DIR/results.json"
echo " Report   : $SCRIPT_DIR/benchmark_report.md"
echo " Ckpt BF16: $SCRIPT_DIR/ckpt_bf16.pt"
echo " Ckpt FP4 : $SCRIPT_DIR/ckpt_nvfp4.pt"
echo "============================================================"
