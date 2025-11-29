#!/usr/bin/env bash
# scripts/run_phase6_experiments.sh
# Phase 6 — Required Experiments (A–F) for the report.
# ======================================================================
# This script:
#   - Defines canonical grids
#   - Runs Baselines (Ridge/RF), GIN, MPNN
#   - Saves runs under runs/, tagging each experiment
#   - Calls the CSV aggregator at the end to produce summary tables
#
# You can selectively run a subset via:
#   bash scripts/run_phase6_experiments.sh A
#   bash scripts/run_phase6_experiments.sh B C
#
# Notes:
# - This script relies on your Hydra CLIs already working (from earlier phases).
# - ESOL is used by default; change DATASET if needed.
# - Exp C (MPNN w/wo edges) requires an option to drop edge_attr. If your
#   datamodule lacks a switch, we proxy “no-edge” by GIN (documented in report).

set -Eeuo pipefail

# ----------------------------- knobs ----------------------------------
DATASET="ESOL"
SEEDS=(0 1 2 3 4)
SPLITS=(scaffold random)

# Baseline (Morgan) default params
MORGAN_BITS=1024
MORGAN_RADIUS=2

# GNN defaults
EPOCHS=50
BATCH=256
HIDDEN=128
ACT="relu"
POOL="mean"
SCHED="plateau"
LR=1e-3
WD=1e-4
NUM_WORKERS=4

# Depth ablation
LAYERS=(2 4 6 8)

# Regularization ablation
DROPOUTS=(0.0 0.1 0.3)
WEIGHT_DECAYS=(0.0 1e-5 1e-4)

# Calibration (F): Enable eval-time MC Dropout by training with dropout>0
# and evaluating with the same model; we tag runs for later reliability plots.
CALIB_DROPOUT=0.1
CALIB_SEEDS=(0 1 2)

# Where to stash *baseline* CSVs (baseline CLI writes under runtime.out_dir)
BASE_OUT_ROOT="outputs/phase6"
mkdir -p "$BASE_OUT_ROOT"

# Helper: stamp + echo
ts() { date +"%Y%m%d-%H%M%S"; }
log() { printf "\n[%s] %s\n" "$(ts)" "$*"; }

# Helper: common train flags
train_common() {
  # emits a list of flags for src.cli.train
  echo \
    "data.name=${DATASET}" \
    "train.epochs=${EPOCHS}" \
    "train.batch_size=${BATCH}" \
    "train.num_workers=${NUM_WORKERS}" \
    "train.lr=${LR}" \
    "train.weight_decay=${WD}" \
    "train.scheduler=${SCHED}" \
    "model.hidden_dim=${HIDDEN}" \
    "model.act=${ACT}" \
    "model.pool=${POOL}" \
    "runtime.quiet=true"
}

# Helper: baseline common flags
baseline_common() {
  echo \
    "data.name=${DATASET}" \
    "feat.kind=morgan" \
    "feat.morgan_bits=${MORGAN_BITS}" \
    "feat.morgan_radius=${MORGAN_RADIUS}" \
    "feat.n_jobs=0" \
    "train.standardize_targets=true" \
    "runtime.quiet=true"
}

# ============================================================================
# A) Split strategy: Compare Random vs. Scaffold for Baselines, GIN, MPNN (5 seeds)
# ============================================================================
run_exp_A() {
  log "Exp A: Split Strategy — Baselines (Ridge/RF), GIN, MPNN across seeds ${SEEDS[*]} and splits ${SPLITS[*]}"

  for split in "${SPLITS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      # Baselines: Ridge
      out_dir="${BASE_OUT_ROOT}/A_split/${DATASET}/baseline_ridge/${split}/seed_${seed}"
      mkdir -p "$out_dir"
      log "A | Baseline Ridge | split=${split} seed=${seed}"
      python -m src.cli.baseline \
        $(baseline_common) \
        "data.split=${split}" \
        "model.name=ridge" \
        "runtime.out_dir=${out_dir}"

      # Baselines: Random Forest (keep it modest)
      out_dir="${BASE_OUT_ROOT}/A_split/${DATASET}/baseline_rf/${split}/seed_${seed}"
      mkdir -p "$out_dir"
      log "A | Baseline RF | split=${split} seed=${seed}"
      python -m src.cli.baseline \
        $(baseline_common) \
        "data.split=${split}" \
        "model.name=random_forest" \
        "model.rf_n_estimators=400" \
        "model.rf_max_depth=None" \
        "runtime.out_dir=${out_dir}"

      # GIN
      log "A | GIN | split=${split} seed=${seed}"
      python -m src.cli.train \
        $(train_common) \
        "eval=${split}" \
        "model.name=gin" \
        "model.num_layers=5" \
        "seed=${seed}"

      # MPNN (requires edge features; your datamodule provides them)
      log "A | MPNN | split=${split} seed=${seed}"
      python -m src.cli.train \
        $(train_common) \
        "eval=${split}" \
        "model.name=mpnn" \
        "model.num_layers=6" \
        "seed=${seed}"
    done
  done
}



# ============================================================================
# B) Depth ablation: layers ∈ {2,4,6,8} for GIN/MPNN; (track runtime via summary.json)
# ============================================================================
run_exp_B() {
  log "Exp B: Depth ablation — LAYERS={${LAYERS[*]}} for GIN & MPNN (seeds: ${SEEDS[*]}, split=scaffold)"
  local split="scaffold"
  for L in "${LAYERS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      # GIN
      log "B | GIN | L=${L} seed=${seed}"
      python -m src.cli.train \
        $(train_common) \
        "eval=${split}" \
        "model.name=gin" \
        "model.num_layers=${L}" \
        "seed=${seed}"

      # MPNN (keep consistent depth)
      python -m src.cli.train \
        $(train_common) \
        "eval=${split}" \
        "model.name=mpnn" \
        "model.num_layers=${L}" \
        "seed=${seed}"
    done
  done
}

# ============================================================================
# C) Edge features ablation: MPNN with/without bond attributes
# ----------------------------------------------------------------------------
# If your datamodule exposes a switch to drop edge_attr (e.g., data.use_edge_attr=false),
# uncomment *_NOEDGE flags below. Otherwise, we *proxy* “no-edge” with GIN (document this).
# ============================================================================
run_exp_C() {
  log "Exp C: Edge features ablation — MPNN (with edges) vs no-edge proxy (GIN). Split=scaffold; seeds=${SEEDS[*]}"

  local split="scaffold"

  for seed in "${SEEDS[@]}"; do
    # With edges: MPNN (standard)
    log "C | MPNN (with edges) | seed=${seed}"
    python -m src.cli.train \
      $(train_common) \
      "eval=${split}" \
      "model.name=mpnn" \
      "model.num_layers=6" \
      "seed=${seed}"
    # Without edges:
    # Preferred (if available):
    #   python -m src.cli.train $(train_common) "eval=${split}" "model=mpnn" "data.use_edge_attr=false" "seed=${seed}"
    # Proxy: use GIN (node-only) to quantify ΔMAE due to edges
    log "C | No-edge proxy → GIN | seed=${seed}"
    python -m src.cli.train \
      $(train_common) \
      "eval=${split}" \
      "model.name=gin" \
      "model.num_layers=6" \
      "seed=${seed}"
  done

  log "C NOTE: If your datamodule supports dropping edge_attr, replace the no-edge proxy (GIN) with MPNN+flag."
}

# ============================================================================
# D) Regularization: Dropout {0.0, 0.1, 0.3} × Weight decay {0, 1e-5, 1e-4}
#     Run on both GIN and MPNN (scaffold) for seeds
# ============================================================================
run_exp_D() {
  log "Exp D: Regularization grid — Dropouts={${DROPOUTS[*]}} × WDs={${WEIGHT_DECAYS[*]}}; split=scaffold"
  local split="scaffold"

  for d in "${DROPOUTS[@]}"; do
    for wd in "${WEIGHT_DECAYS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        # GIN
        log "D | GIN | dropout=${d} wd=${wd} seed=${seed}"
        python -m src.cli.train \
          $(train_common) \
          "eval=${split}" \
          "model.name=gin" \
          "model.num_layers=6" \
          "train.weight_decay=${wd}" \
          "model.dropout=${d}" \
          "seed=${seed}"

        # MPNN
        log "D | MPNN | dropout=${d} wd=${wd} seed=${seed}"
        python -m src.cli.train \
          $(train_common) \
          "eval=${split}" \
          "model.name=mpnn" \
          "model.num_layers=6" \
          "train.weight_decay=${wd}" \
          "model.dropout=${d}" \
          "seed=${seed}"
      done
    done
  done
}

# ============================================================================
# E) Feature ablation (nodes): remove aromaticity / formal charge; re-train best depth
# ----------------------------------------------------------------------------
# This requires your datamodule to offer toggles (e.g., data.node_feats.drop=[aromaticity]).
# If not present, leave comments (and proxy by training as-is; then update once flags exist).
# ============================================================================
run_exp_E() {
  log "Exp E: Node feature ablation — requires datamodule toggles. Using proxies/documentation if absent."
  local split="scaffold"
  local best_L=6

  # Example (uncomment/change if your datamodule supports these):
  #   FEAT_DROPS=("aromaticity" "formal_charge")
  # for feat in "${FEAT_DROPS[@]}"; do
  #   for seed in "${SEEDS[@]}"; do
  #     log "E | GIN drop=${feat} | seed=${seed}"
  #     python -m src.cli.train \
  #       $(train_common) \
  #       "eval=${split}" \
  #       "model=gin" \
  #       "model.num_layers=${best_L}" \
  #       "data.node_feats.drop=[${feat}]" \
  #       "seed=${seed}"
  #     log "E | MPNN drop=${feat} | seed=${seed}"
  #     python -m src.cli.train \
  #       $(train_common) \
  #       "eval=${split}" \
  #       "model=mpnn" \
  #       "model.num_layers=${best_L}" \
  #       "data.node_feats.drop=[${feat}]" \
  #       "seed=${seed}"
  #   done
  # done

  # Proxy: run “no-ablation” to keep the directory structure predictable; update once flags exist.
  for seed in "${SEEDS[@]}"; do
    log "E | Proxy (no ablation) GIN | seed=${seed}"
    python -m src.cli.train \
      $(train_common) \
      "eval=${split}" \
      "model.name=gin" \
      "model.num_layers=${best_L}" \
      "seed=${seed}"

    log "E | Proxy (no ablation) MPNN | seed=${seed}"
    python -m src.cli.train \
      $(train_common) \
      "eval=${split}" \
      "model.name=mpnn" \
      "model.num_layers=${best_L}" \
      "seed=${seed}"
  done

  log "E NOTE: Replace proxies with real feature-drop flags once available in the datamodule."
}

# ============================================================================
# F) Calibration: reliability plots + ECE (with/without MC dropout)
# ----------------------------------------------------------------------------
# We generate two tagged runs (dropout=0.0 vs 0.1), across seeds.
# Reliability/ECE figures are produced later by your analysis notebooks/scripts.
# ============================================================================
run_exp_F() {
  log "Exp F: Calibration — with/without MC-dropout (dropout=${CALIB_DROPOUT}); seeds=${CALIB_SEEDS[*]}"
  local split="scaffold"

  for seed in "${CALIB_SEEDS[@]}"; do
    # No-dropout baseline
    log "F | GIN no-dropout | seed=${seed}"
    python -m src.cli.train \
      $(train_common) \
      "eval=${split}" \
      "model.name=gin" \
      "model.num_layers=6" \
      "model.dropout=0.0" \
      "seed=${seed}"

    # With dropout
    log "F | GIN dropout=${CALIB_DROPOUT} | seed=${seed}"

    python -m src.cli.train \
      $(train_common) \
      "eval=${split}" \
      "model.name=gin" \
      "model.num_layers=6" \
      "model.dropout=${CALIB_DROPOUT}" \
      "seed=${seed}"


    # MPNN — mirrored
    log "F | MPNN no-dropout | seed=${seed}"
    python -m src.cli.train \
      $(train_common) \
      "eval=${split}" \
      "model.name=mpnn" \
      "model.num_layers=6" \
      "model.dropout=0.0" \
      "seed=${seed}"

    log "F | MPNN dropout=${CALIB_DROPOUT} | seed=${seed}"
    python -m src.cli.train \
      $(train_common) \
      "eval=${split}" \
      "model.name=mpnn" \
      "model.num_layers=6" \
      "model.dropout=${CALIB_DROPOUT}" \
      "seed=${seed}"
  done

  log "F NOTE: Use your Phase-5 metrics utilities to compute ECE & build reliability diagrams from predictions."
}

# ============================================================================
# Aggregation step (optional but recommended)
# Produces a summary CSV under report/tables/summary.csv by sweeping runs/
# ============================================================================
aggregate_all() {
  log "Aggregating CSVs into report tables…"
  # Aggregate everything under runs/ plus the baseline outputs
  python -m src.analysis.aggregate_csv \
    --roots "runs" "${BASE_OUT_ROOT}" \
    --out "report/tables/summary.csv" \
    --group-by dataset model split \
    --metrics test_MAE test_RMSE val_MAE val_RMSE
  log "Wrote report/tables/summary.csv"
}



# ----------------------------- driver ---------------------------------
main() {
  local wanted=("$@")
  if ((${#wanted[@]}==0)); then
    # Run everything if no args provided
    wanted=(A B C D E F)
  fi

  for exp in "${wanted[@]}"; do
    case "$exp" in
      A|a) run_exp_A ;;
      B|b) run_exp_B ;;
      C|c) run_exp_C ;;
      D|d) run_exp_D ;;
      E|e) run_exp_E ;;
      F|f) run_exp_F ;;
      *) echo "Unknown experiment key: $exp (use A B C D E F)"; exit 1 ;;
    esac
  done

  aggregate_all
  log "Phase 6 batch complete."
}

main "$@"
