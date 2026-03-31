#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_fp_benchmark.sh  —  FP16 vs INT8-PTQ vs FP8-Native benchmark
#
# Usage:
#   cd /mnt/c/MachineLearning/UsamaKenway/nvfp4_/our_solution
#   bash run_fp_benchmark.sh
#
# Architecture: 12-layer · 12-head · 768-dim GPT (GPT-2 scale)
# Iterations:   5 000
#
# Output files:
#   ckpt_fp16.pt           FP16 training checkpoint
#   ckpt_fp8.pt            FP8 training checkpoint
#   results_fp.json        Raw metrics (all phases)
#   fp_benchmark_report.md Final comparison table + analysis
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../nanoGPT/data/shakespeare_char"

echo "============================================================"
echo " FP16 vs INT8-PTQ vs FP8-Native Benchmark"
echo " Model: 12L · 12H · 768D (GPT-2 scale)"
echo " Iters: 5 000"
echo "============================================================"
echo ""

if [ ! -f "$DATA_DIR/train.bin" ]; then
    echo "[0/4] Preparing shakespeare_char dataset…"
    python "$DATA_DIR/prepare.py"
else
    echo "[0/4] Dataset ready."
fi
echo ""

echo "[1/4] Phase 1: FP16 baseline training (5 000 iters)…"
python "$SCRIPT_DIR/train_fp16.py"
echo ""

echo "[2/4] Phase 2: INT8 Post-Training Quantisation…"
python "$SCRIPT_DIR/quantize_int8.py"
echo ""

echo "[3/4] Phase 3: Native FP8 training (5 000 iters)…"
python "$SCRIPT_DIR/train_fp8.py"
echo ""

echo "[4/4] Phase 4: Generating report…"
python "$SCRIPT_DIR/generate_fp_report.py"
echo ""

echo "============================================================"
echo " Done!"
echo " Results : $SCRIPT_DIR/results_fp.json"
echo " Report  : $SCRIPT_DIR/fp_benchmark_report.md"
echo "============================================================"
