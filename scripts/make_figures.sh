#!/usr/bin/env bash
# scripts/make_figures.sh
# Phase 9 — Plots & tables pipeline
# ----------------------------------------------------------------------
# This script calls src.analysis.plots to generate:
#   * Depth curves (GIN & MPNN, ESOL, scaffold)
#   * Regularization ablation bars (dropout × weight_decay)
#   * Calibration curves (if a calibration CSV is present)
#
# Definition of Done:
#   - After running this script from the repo root, you should find:
#       report/figures/esol_scaffold_depth_mae.png
#       report/figures/esol_scaffold_depth_rmse.png
#       report/figures/esol_scaffold_ablation_gin.png
#       report/figures/esol_scaffold_ablation_mpnn.png
#       report/figures/calibration_reliability.png  (if calibration CSV exists)

set -Eeuo pipefail

ROOTS=("runs" "outputs/phase6")
OUTDIR="report/figures"
CALIB_CSV="report/tables/calibration_bins.csv"

mkdir -p "${OUTDIR}"

echo "[make_figures] Using roots: ${ROOTS[*]}"
echo "[make_figures] Writing figures to: ${OUTDIR}"

# 1) Depth curves
echo "[make_figures] Generating depth curves (GIN & MPNN, ESOL, scaffold)..."
python -m src.analysis.plots depth \
  --roots "${ROOTS[@]}" \
  --dataset ESOL \
  --split scaffold \
  --outdir "${OUTDIR}"

# 2) Regularization ablation bars
echo "[make_figures] Generating regularization ablation bars..."
python -m src.analysis.plots ablation \
  --roots "${ROOTS[@]}" \
  --dataset ESOL \
  --split scaffold \
  --outdir "${OUTDIR}"

# 3) Calibration curves (optional)
if [[ -f "${CALIB_CSV}" ]]; then
  echo "[make_figures] Found ${CALIB_CSV}; generating calibration curves..."
  python -m src.analysis.plots calibration \
    --calibration-csv "${CALIB_CSV}" \
    --outdir "${OUTDIR}"
else
  echo "[make_figures] No ${CALIB_CSV} found; skipping calibration curves."
fi

echo "[make_figures] Done. Figures are available under ${OUTDIR}"
