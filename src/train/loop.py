from __future__ import annotations
"""
Unified training loop for molecular GNNs (GIN / MPNN)

- ESOL / QM9 via our datamodules
- GIN or MPNN models
- Target standardization (optional) with de-standardized metrics
- EMA (optional)
- Mixed precision (optional; CUDA only)
- Early stopping on val MAE + best.ckpt
- Schedulers: none | cosine | cosine_warmup | plateau
"""

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, make_dataclass, field as _dc_field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# Our modules
from src.models.gin import GINConfig, GINNet
from src.models.mpnn import MPNNConfig, MPNNNet
from src.datamodules import esol as dm_esol
from src.datamodules import qm9 as dm_qm9

# --- add this helper right after the dm_* imports ---
from typing import Any as _Any, Optional as _Optional

def _namespace_to_dataclass(ns) -> object:
    """Convert a SimpleNamespace (or any mapping) to a dynamic dataclass instance,
    so downstream code that calls `dataclasses.asdict(cfg)` works unchanged."""
    d = dict(ns.__dict__) if hasattr(ns, "__dict__") else dict(ns)
    fields = []
    for k, v in d.items():
        tp = type(v) if v is not None else _Optional[_Any]
        fields.append((k, tp, _dc_field(default=v)))
    RuntimeCfg = make_dataclass("RuntimeCfg", fields)
    return RuntimeCfg(**d)
# --- end helper ---


# -------------------------------------------------------
# Utilities
# -------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


@torch.no_grad()
def compute_mae_rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, float]:
    err = (y_true - y_pred).view(-1)
    mae = float(err.abs().mean().item())
    rmse = float(torch.sqrt((err ** 2).mean()).item())
    return mae, rmse


# -------------------------------------------------------
# Exponential Moving Average (EMA)
# -------------------------------------------------------

class EMA:
    """Exponential moving average of model parameters (stochastic weight averaging style)."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self.register(model)

    def register(self, model: nn.Module):
        self.shadow.clear()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            new_avg = (1.0 - d) * param.detach() + d * self.shadow[name]
            self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model: nn.Module):
        self.backup.clear()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name].data)

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data.copy_(self.backup[name].data)
        self.backup.clear()


# -------------------------------------------------------
# Data plumbing
# -------------------------------------------------------

def _guess_dims_from_loader(loader: DataLoader) -> Tuple[int, Optional[int]]:
    """Peek a single batch to infer node_dim and (optional) edge_dim."""
    for batch in loader:
        x_dim = int(batch.x.size(-1))
        e_dim = int(batch.edge_attr.size(-1)) if hasattr(batch, "edge_attr") and batch.edge_attr is not None else None
        return x_dim, e_dim
    raise RuntimeError("Empty loader; cannot infer feature dims.")


def build_dataloaders(args) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Construct train/val/test loaders and return metadata dict with dataset info.
    Assumes ESOL/QM9 datamodules expose `make_dataloaders(cfg_like)`.
    """
    common = SimpleNamespace(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=None,
        split=args.split,              # expected by our datamodules
        limit_n=args.limit_n,
        pin_memory=args.pin_memory,
        persistent_workers=(not args.no_persistent_workers) and args.num_workers > 0,
        drop_last=args.drop_last,
        shuffle_train=(not args.no_shuffle_train),
        verbose=(not args.quiet),
    )
    if args.dataset.upper() == "ESOL":
        train_loader, val_loader, test_loader, meta = dm_esol.make_dataloaders(_namespace_to_dataclass(common))
    elif args.dataset.upper() == "QM9":
        # add qm9 target selection hints
        qm9_cfg = SimpleNamespace(**common.__dict__)
        qm9_cfg.target_key = args.target
        qm9_cfg.target_index = args.target_index
        train_loader, val_loader, test_loader, meta = dm_qm9.make_dataloaders(_namespace_to_dataclass(qm9_cfg))
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    return train_loader, val_loader, test_loader, meta


# -------------------------------------------------------
# Model factory
# -------------------------------------------------------

def build_model(args, node_dim: int, edge_dim: Optional[int]) -> nn.Module:
    if args.model.lower() == "gin":
        cfg = GINConfig(
            in_dim=node_dim,
            edge_dim=edge_dim,
            out_dim=1,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_norm=not args.no_batch_norm,
            residual=not args.no_residual,
            act=args.act,
            pool=args.pool,
            use_edge_attr=args.use_edge_attr and (edge_dim is not None),
            learn_eps=not args.no_learn_eps,
            virtual_node=args.virtual_node,
            readout_layers=args.readout_layers,
            readout_hidden_mult=args.readout_hidden_mult,
            init=args.init,
        )
        model = GINNet(cfg)
    elif args.model.lower() == "mpnn":
        if edge_dim is None:
            raise ValueError("MPNN requires edge_attr (edge_dim is None from datamodule batch).")
        cfg = MPNNConfig(
            in_dim=node_dim,
            edge_dim=edge_dim,
            out_dim=1,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            aggr=args.aggr,
            pool=args.pool,
            act=args.act,
            dropout=args.dropout,
            batch_norm=not args.no_batch_norm,
            residual=not args.no_residual,
            use_gru=not args.no_gru,  # flag is "no_gru" to disable
            readout_layers=args.readout_layers,
            readout_hidden_mult=args.readout_hidden_mult,
            init=args.init,
            verbose=True,
        )
        model = MPNNNet(cfg)
    else:
        raise ValueError(f"Unknown model {args.model}")
    return model


# -------------------------------------------------------
# Target normalization helpers
# -------------------------------------------------------

@torch.no_grad()
def _compute_target_stats(loader: DataLoader, device) -> dict:
    vals = []
    for batch in loader:
        y = batch.y.view(-1, 1).to(device, non_blocking=True)
        vals.append(y)
    y = torch.cat(vals, dim=0).float()
    mean = y.mean().item()
    std = y.std(unbiased=False).item()
    if std <= 0:
        std = 1.0
    return {"mean": float(mean), "std": float(std)}

def _standardize_targets(y: torch.Tensor, stats: dict | None):
    if stats is None:
        return y
    return (y - stats["mean"]) / stats["std"]

def _destandardize_preds(p: torch.Tensor, stats: dict | None):
    if stats is None:
        return p
    return p * stats["std"] + stats["mean"]


# -------------------------------------------------------
# Train/Eval loops
# -------------------------------------------------------

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, device, scaler, amp: bool,
                    ema: Optional[EMA] = None, target_stats: dict | None = None):
    model.train()
    loss_fn = nn.MSELoss(reduction="mean")
    running = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        y = batch.y.view(-1, 1).to(device)
        y_std = _standardize_targets(y, target_stats)

        optimizer.zero_grad(set_to_none=True)

        if amp:
            with torch.cuda.amp.autocast():
                pred = model(batch)
                loss = loss_fn(pred, y_std)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(batch)
            loss = loss_fn(pred, y_std)
            loss.backward()
            optimizer.step()

        if ema is not None:
            ema.update(model)

        running += float(loss.item()) * y.size(0)
        n += y.size(0)
    return running / max(n, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device, amp: bool,
             ema: Optional[EMA] = None, target_stats: dict | None = None) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.MSELoss(reduction="mean")

    total_loss = 0.0
    all_y: List[torch.Tensor] = []
    all_p: List[torch.Tensor] = []
    n = 0

    using_ema = False
    if ema is not None:
        ema.apply_shadow(model)
        using_ema = True

    try:
        for batch in loader:
            batch = batch.to(device)
            y = batch.y.view(-1, 1).to(device)
            y_std = _standardize_targets(y, target_stats)
            if amp:
                with torch.cuda.amp.autocast():
                    pred = model(batch)
                    loss = loss_fn(pred, y_std)
            else:
                pred = model(batch)
                loss = loss_fn(pred, y_std)

            all_y.append(y.detach().cpu())
            all_p.append(pred.detach().cpu())
            total_loss += float(loss.item()) * y.size(0)
            n += y.size(0)

        y_true = torch.cat(all_y, dim=0)
        y_pred_std = torch.cat(all_p, dim=0)
        y_pred = _destandardize_preds(y_pred_std, target_stats)
        mae, rmse = compute_mae_rmse(y_true, y_pred)
        return {"MSE": total_loss / max(n, 1), "MAE": mae, "RMSE": rmse}
    finally:
        if using_ema:
            ema.restore(model)


# -------------------------------------------------------
# Training orchestration
# -------------------------------------------------------

def run(args) -> Dict[str, Any]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ensure_dir(args.out_dir)

    # Data
    train_loader, val_loader, test_loader, meta = build_dataloaders(args)
    node_dim, edge_dim = _guess_dims_from_loader(train_loader)
    target_stats = _compute_target_stats(train_loader, device) if args.standardize_targets else None

    # Model
    model = build_model(args, node_dim=node_dim, edge_dim=edge_dim).to(device)
    n_params = count_parameters(model, trainable_only=True)
    print(f"[loop] Model: {args.model.upper()} | params: {n_params:,d} | device: {device}")

    # Optimizer & (optional) scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    elif args.scheduler == "cosine_warmup":
        warmup_epochs = max(int(args.warmup_epochs), 0)
        if warmup_epochs > 0:
            warm = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
            cos = CosineAnnealingLR(optimizer, T_max=max(args.epochs - warmup_epochs, 1))
            scheduler = SequentialLR(optimizer, schedulers=[warm, cos], milestones=[warmup_epochs])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(args.patience // 2, 1)
        )
    else:
        scheduler = None

    # Mixed precision scaler, EMA
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)  # new API
    ema = EMA(model, decay=args.ema_decay) if args.ema else None

    # Early stopping and checkpoint
    best_val = float("inf")
    best_epoch = -1
    patience_left = int(args.patience)
    ckpt_path = os.path.join(args.out_dir, "best.ckpt")

    history: List[Dict[str, Any]] = []

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        # Train
        train_mse = train_one_epoch(
            model, train_loader, optimizer, device, scaler, use_amp, ema=ema, target_stats=target_stats
        )

        # Val
        val_metrics = evaluate(model, val_loader, device, amp=use_amp, ema=ema, target_stats=target_stats)
        val_mae = val_metrics["MAE"]

        # Scheduler step
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_mae)
            else:
                scheduler.step()

        # Track
        rec = {
            "epoch": epoch,
            "train/MSE": train_mse,
            "val/MSE": val_metrics["MSE"],
            "val/MAE": val_metrics["MAE"],
            "val/RMSE": val_metrics["RMSE"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(rec)

        print(f"[loop] Epoch {epoch:03d} | "
              f"train MSE {train_mse:.4f} | val MAE {val_metrics['MAE']:.4f} | "
              f"val RMSE {val_metrics['RMSE']:.4f} | lr {rec['lr']:.2e}")

        # Early stopping on val MAE (minimize)
        improved = val_mae < (best_val - args.min_delta)
        if improved:
            best_val = val_mae
            best_epoch = epoch
            patience_left = int(args.patience)  # reset
            # Save checkpoint with either EMA or raw weights? Save raw but store ema shadow too
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_mae": best_val,
                "args": vars(args),
                "meta": meta,
            }
            if ema is not None:
                state["ema_shadow"] = {k: v.cpu() for k, v in ema.shadow.items()}
            torch.save(state, ckpt_path)
            print(f"[loop]   ↳ New best! Saved checkpoint to {ckpt_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[loop] Early stopping at epoch {epoch} (best epoch {best_epoch}, best val MAE {best_val:.4f})")
                break

    # Load best for test evaluation (use EMA during eval if provided)
    if os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            # Older PyTorch without `weights_only` arg
            state = torch.load(ckpt_path, map_location="cpu")
        
        model.load_state_dict(state["model"])
        if ema is not None and "ema_shadow" in state:
            # Refresh EMA shadow from checkpoint so test uses the best EMA
            for k, v in state["ema_shadow"].items():
                ema.shadow[k] = v.to(device)
    else:
        print("[loop] WARNING: best.ckpt not found; evaluating last epoch weights.")

    test_metrics = evaluate(model, test_loader, device, amp=use_amp, ema=ema, target_stats=target_stats)
    elapsed = time.time() - start_time

    # Save metrics CSV
    ensure_dir(args.out_dir)
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    header = [
        "dataset","model","split","epochs","best_epoch","seed",
        "val_MAE","val_RMSE","test_MAE","test_RMSE",
        "lr","weight_decay","batch_size","hidden_dim","num_layers",
        "ema","ema_decay","amp","scheduler","elapsed_sec",
    ]
    row = [
        args.dataset.upper(), args.model.lower(), args.split, args.epochs, best_epoch, args.seed,
        best_val, next((r["val/RMSE"] for r in history if r["epoch"] == best_epoch), float("nan")),
        test_metrics["MAE"], test_metrics["RMSE"],
        args.lr, args.weight_decay, args.batch_size, args.hidden_dim, args.num_layers,
        int(args.ema), args.ema_decay, int(use_amp), args.scheduler, round(elapsed, 2),
    ]
    write_header = (not os.path.exists(csv_path))
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

    # Save summary JSON
    summary = {
        "best_epoch": best_epoch,
        "best_val_MAE": best_val,
        "test": test_metrics,
        "elapsed_sec": elapsed,
        "params": n_params,
        "dataset_meta": meta,
        "history_last": history[-5:] if history else [],
        "paths": {"best_ckpt": ckpt_path, "metrics_csv": csv_path},
        "target_norm": target_stats,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[loop] Done. Best@{best_epoch}  val MAE={best_val:.4f} | "
          f"test MAE={test_metrics['MAE']:.4f}, RMSE={test_metrics['RMSE']:.4f}")
    print(f"[loop] Saved: {ckpt_path}, {csv_path}")
    return summary


# -------------------------------------------------------
# CLI
# -------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train molecular GNNs with early stopping and EMA.")
    # Data
    p.add_argument("--dataset", type=str, default="ESOL", choices=["ESOL","QM9"])
    p.add_argument("--root", type=str, default="data")
    p.add_argument("--split", type=str, default="scaffold", choices=["scaffold","random"])
    p.add_argument("--limit-n", type=int, default=None, help="QM9 subset; None/full if omitted")
    p.add_argument("--target", type=str, default="U0", help="QM9 target key (ignored for ESOL)")
    p.add_argument("--target-index", type=int, default=None, help="QM9 target index (overrides key)")

    # Model
    p.add_argument("--model", type=str, default="gin", choices=["gin","mpnn"])
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--pool", type=str, default="mean", choices=["sum","mean","max"])
    p.add_argument("--act", type=str, default="relu", choices=["relu","gelu","leaky_relu","elu"])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-batch-norm", action="store_true")
    p.add_argument("--no-residual", action="store_true")
    p.add_argument("--use-edge-attr", action="store_true", help="Enable GINE path if edge_attr exists")
    p.add_argument("--no-learn-eps", action="store_true", help="GIN: disable learnable epsilon")
    p.add_argument("--virtual-node", action="store_true", help="GIN: enable virtual node")
    p.add_argument("--readout-layers", type=int, default=2)
    p.add_argument("--readout-hidden-mult", type=float, default=1.0)
    p.add_argument("--aggr", type=str, default="add", choices=["add","mean","max"], help="MPNN only")
    p.add_argument("--no-gru", action="store_true", help="MPNN: disable GRU updates")
    p.add_argument("--init", type=str, default="kaiming", choices=["kaiming","xavier","none"])

    # Optimization
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, default="plateau", choices=["none","cosine","cosine_warmup","plateau"])

    # Dataloader behavior
    p.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory.")
    p.add_argument("--no-persistent-workers", action="store_true", help="Disable persistent_workers in DataLoader.")
    p.add_argument("--drop-last", action="store_true", help="Enable drop_last on DataLoaders (train/val/test).")
    p.add_argument("--no-shuffle-train", action="store_true", help="Disable shuffling on the training DataLoader.")

    # Early stopping
    p.add_argument("--patience", type=int, default=10, help="epochs without val MAE improvement")
    p.add_argument("--min-delta", type=float, default=1e-4, help="minimum MAE improvement to reset patience")

    # EMA + AMP
    p.add_argument("--ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--amp", action="store_true")

    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out-dir", type=str, default="runs/exp")
    p.add_argument("--quiet", action="store_true", help="Reduce datamodule/model logging.")

    # Split ratios
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)

    # Target standardization + warmup
    p.add_argument("--standardize-targets", action="store_true",
                   help="Train on z-scored targets and report metrics on original scale.")
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="Linear warmup epochs for cosine_warmup scheduler.")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
