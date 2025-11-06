# src/cli/train.py
from __future__ import annotations

"""
Hydra-powered training CLI for molecular GNNs.

Usage examples:
  # Minimal: ESOL + GIN, scaffold split
  python -m src.cli.train data.name=ESOL eval=scaffold model=gin seed=0

  # QM9 + MPNN subset with target standardization and cosine warmup
  python -m src.cli.train \
    data.name=QM9 data.limit_n=5000 data.target=U0 eval=random \
    model=mpnn train.standardize_targets=true train.scheduler=cosine_warmup train.warmup_epochs=2 \
    train.epochs=30 train.batch_size=256 seed=7

Notes:
  - `eval` is an alias for `split` to match the DoD example.
  - We always resolve `out_dir` relative to the original working directory (Hydra can change CWD).
  - All flags are mapped to `src.train.loop` to keep a single source of truth for actual training.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, List

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, DictConfig

from src.train.loop import build_argparser, run as run_loop


# -----------------------------------------------------------------------------
# Config schema (Hydra)
# -----------------------------------------------------------------------------

@dataclass
class DataCfg:
    name: str = "ESOL"           # ESOL | QM9
    root: str = "data"
    limit_n: Optional[int] = None  # QM9 subset size (None for full)
    # QM9 target selection
    target: str = "U0"
    target_index: Optional[int] = None


@dataclass
class ModelCfg:
    # model core
    name: str = "gin"  # gin | mpnn

    # shared
    hidden_dim: int = 128
    num_layers: int = 5
    pool: str = "mean"  # sum | mean | max
    act: str = "relu"   # relu | gelu | leaky_relu | elu
    dropout: float = 0.10
    init: str = "kaiming"  # kaiming | xavier | none

    # normalization / residual
    no_batch_norm: bool = False
    no_residual: bool = False

    # GIN/GINE specifics
    use_edge_attr: bool = False
    no_learn_eps: bool = False
    virtual_node: bool = False
    readout_layers: int = 2
    readout_hidden_mult: float = 1.0

    # MPNN specifics
    aggr: str = "add"       # add | mean | max
    no_gru: bool = False    # True disables GRU updates in MPNN


@dataclass
class TrainCfg:
    # optimization
    epochs: int = 50
    batch_size: int = 256
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "plateau"  # none | cosine | cosine_warmup | plateau

    # dataloader behavior
    pin_memory: bool = False
    persistent_workers: bool = True
    drop_last: bool = False
    shuffle_train: bool = True

    # early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # target normalization
    standardize_targets: bool = False
    warmup_epochs: int = 0  # for cosine_warmup

    # EMA + AMP
    ema: bool = False
    ema_decay: float = 0.999
    amp: bool = False

    # Calibration (classification-only; no-op for regression unless --is-classification is set)
    p.add_argument("--is-classification", action="store_true",
                   help="Enable classification path (collect logits/labels, accuracy/NLL/ECE, and optional calibration).")
    p.add_argument("--calibration-kind", type=str, default="none",
                   choices=["none","temperature","vector","matrix","dirichlet_simple","dirichlet_full","isotonic"],
                   help="Post-hoc calibrator to fit on validation logits.")
    p.add_argument("--calibration-optimizer", type=str, default="lbfgs", choices=["lbfgs","adam"])
    p.add_argument("--calibration-lr", type=float, default=0.1)
    p.add_argument("--calibration-max-iters", type=int, default=300)
    p.add_argument("--calibration-l2", type=float, default=1e-4, help="Regularization for matrix/dirichlet calibrators.")
    p.add_argument("--calibration-verbose", action="store_true")

    


@dataclass
class RuntimeCfg:
    seed: int = 0
    cpu: bool = False
    out_dir: Optional[str] = None
    quiet: bool = False

    # split ratios
    train_frac: float = 0.8
    val_frac: float = 0.1


@dataclass
class RootCfg:
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)
    calib: CalibCfg = field(default_factory=CalibCfg) 
    # Alias for split strategy to match DoD (eval=scaffold/random)
    eval: str = "scaffold"  # scaffold | random

    seed: Optional[int] = None
# Register config with Hydra so @hydra.main can instantiate it
cs = ConfigStore.instance()
cs.store(name="train_config", node=RootCfg)


# -----------------------------------------------------------------------------
# Helper: build argv for src.train.loop
# -----------------------------------------------------------------------------

def _compose_loop_argv(cfg: RootCfg, original_cwd: str) -> List[str]:
    """
    Translate Hydra config to the argparse flags of src.train.loop.
    Keeps loop.py as the single source of truth for training behavior.
    """
    # Resolve out_dir (relative to original cwd, not Hydra run dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if cfg.runtime.out_dir is None:
        # e.g., runs/esol_gin_scaffold_seed0_20250101-120000
        tag = f"{cfg.data.name.lower()}_{cfg.model.name.lower()}_{cfg.eval}_seed{cfg.runtime.seed}_{timestamp}"
        out_dir = os.path.join(original_cwd, "runs", tag)
    else:
        out_dir = cfg.runtime.out_dir
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(original_cwd, out_dir)

    args: List[str] = [
        "--dataset", cfg.data.name.upper(),
        "--root", os.path.join(original_cwd, cfg.data.root) if not os.path.isabs(cfg.data.root) else cfg.data.root,
        "--split", cfg.eval,
        "--epochs", str(cfg.train.epochs),
        "--batch-size", str(cfg.train.batch_size),
        "--num-workers", str(cfg.train.num_workers),
        "--lr", str(cfg.train.lr),
        "--weight-decay", str(cfg.train.weight_decay),
        "--scheduler", cfg.train.scheduler,
        "--seed", str(cfg.runtime.seed),
        "--out-dir", out_dir,
        "--train-frac", str(cfg.runtime.train_frac),
        "--val-frac", str(cfg.runtime.val_frac),
        "--model", cfg.model.name.lower(),
        "--hidden-dim", str(cfg.model.hidden_dim),
        "--num-layers", str(cfg.model.num_layers),
        "--pool", cfg.model.pool,
        "--act", cfg.model.act,
        "--dropout", str(cfg.model.dropout),
        "--readout-layers", str(cfg.model.readout_layers),
        "--readout-hidden-mult", str(cfg.model.readout_hidden_mult),
        "--init", cfg.model.init,
        "--aggr", cfg.model.aggr,
        "--target", cfg.data.target,
    ]

    # Optional/boolean flags
    if cfg.data.limit_n is not None:
        args += ["--limit-n", str(cfg.data.limit_n)]
    if cfg.data.target_index is not None:
        args += ["--target-index", str(cfg.data.target_index)]
    if cfg.train.pin_memory:
        args.append("--pin-memory")
    if not cfg.train.persistent_workers:
        args.append("--no-persistent-workers")
    if cfg.train.drop_last:
        args.append("--drop-last")
    if not cfg.train.shuffle_train:
        args.append("--no-shuffle-train")

    # Early stopping
    args += ["--patience", str(cfg.train.patience), "--min-delta", str(cfg.train.min_delta)]

    # Target standardization + warmup
    if cfg.train.standardize_targets:
        args.append("--standardize-targets")
    if cfg.train.scheduler == "cosine_warmup":
        args += ["--warmup-epochs", str(cfg.train.warmup_epochs)]

    # EMA + AMP
    if cfg.train.ema:
        args.append("--ema")
    args += ["--ema-decay", str(cfg.train.ema_decay)]
    if cfg.train.amp:
        args.append("--amp")

    # Runtime
    if cfg.runtime.cpu:
        args.append("--cpu")
    if cfg.runtime.quiet:
        args.append("--quiet")

    # Model boolean flips
    if cfg.model.no_batch_norm:
        args.append("--no-batch-norm")
    if cfg.model.no_residual:
        args.append("--no-residual")
    if cfg.model.use_edge_attr:
        args.append("--use-edge-attr")
    if cfg.model.no_learn_eps:
        args.append("--no-learn-eps")
    if cfg.model.virtual_node:
        args.append("--virtual-node")
    if cfg.model.no_gru:
        args.append("--no-gru")




    # Calibration (classification-only; forwarded to src.train.loop argparse)
    if cfg.calib.is_classification:
        args.append("--is-classification")
    if cfg.calib.calibration_kind and cfg.calib.calibration_kind != "none":
        args += ["--calibration-kind", cfg.calib.calibration_kind]
        args += ["--calibration-optimizer", cfg.calib.calibration_optimizer]
        args += ["--calibration-lr", str(cfg.calib.calibration_lr)]
        args += ["--calibration-max-iters", str(cfg.calib.calibration_max_iters)]
        args += ["--calibration-l2", str(cfg.calib.calibration_l2)]
        if cfg.calib.calibration_verbose:
            args.append("--calibration-verbose")





    return args



# -----------------------------------------------------------------------------
# Shorthand coercions
# -----------------------------------------------------------------------------

def _coerce_model_shorthand(conf: "RootCfg") -> "RootCfg":
    """
    Allow `model=gin` or `model=mpnn` on the CLI by wrapping a bare string
    into our ModelCfg dataclass. If model is already a dataclass/dict, leave it.
    """
    try:
        if isinstance(conf.model, str):
            conf.model = ModelCfg(name=conf.model)
        elif isinstance(conf.model, dict) and "name" in conf.model and not hasattr(conf.model, "__dataclass_fields__"):
            conf.model = ModelCfg(**conf.model)
    except Exception:
        # Leave as-is and let Hydra/argparse raise a useful error
        pass
    return conf




# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

@hydra.main(version_base=None, config_name="train_config")
def main(cfg: DictConfig) -> int:
    # Convert DictConfig to our dataclass RootCfg for type hints / IDE comfort
    conf = OmegaConf.to_object(cfg)  # -> nested dicts/dataclasses (thanks to ConfigStore)
    assert isinstance(conf, RootCfg) or isinstance(conf, dict)

    # If DictConfig -> dataclass, fine; if plain dict, coerce into dataclass
    if isinstance(conf, dict):
        # reconstruct dataclass, but Hydra normally instantiates dataclass already
        conf = RootCfg(**conf)  # type: ignore[call-arg]


    conf = _coerce_model_shorthand(conf)


    # NEW: map top-level seed alias -> runtime.seed
    if getattr(conf, "seed", None) is not None:
        conf.runtime.seed = int(conf.seed)


    original_cwd = get_original_cwd()
    argv = _compose_loop_argv(conf, original_cwd)

    # Delegate to the canonical training loop (single source of truth)
    parser = build_argparser()
    args = parser.parse_args(argv)
    summary = run_loop(args)

    # Print a brief post-run summary (so Hydra logs contain the key outputs)
    out_dir = args.out_dir
    print("\n[cli.train] Done.")
    print(f"[cli.train] Out dir : {out_dir}")
    print(f"[cli.train] Best@{summary.get('best_epoch')} | "
          f"val MAE={summary.get('best_val_MAE'):.4f} | "
          f"test MAE={summary.get('test',{}).get('MAE'):.4f}, "
          f"RMSE={summary.get('test',{}).get('RMSE'):.4f}")
    print(f"[cli.train] Artifacts: {summary.get('paths',{}).get('best_ckpt')}, "
          f"{summary.get('paths',{}).get('metrics_csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
