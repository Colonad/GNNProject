# src/train/sweep.py
from __future__ import annotations

import json
import os
import argparse
from typing import Dict, Any, List

from .loop import run, build_argparser


def _run_single(base_args_list: List[str], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the base CLI args using the training loop's parser, apply Python-level
    overrides (e.g., lr/hidden_dim), run training, and return the summary.
    """
    parser = build_argparser()
    args = parser.parse_args(base_args_list)

    for k, v in overrides.items():
        setattr(args, k, v)

    # Put each trial in its own directory under --out-dir
    tag = f"hd{args.hidden_dim}_lr{args.lr}"
    args.out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(args.out_dir, exist_ok=True)

    summary = run(args)
    summary["trial_overrides"] = overrides
    summary["trial_out_dir"] = args.out_dir
    return summary


def main(argv: List[str] | None = None) -> int:
    """
    Tiny hyperparameter sweep wrapper over src.train.loop.run.

    Example:
        python -m src.train.sweep --dataset ESOL --model gin --split scaffold \
            --epochs 1 --out-dir runs/sweep --hidden-dims 32 48 --lrs 1e-3 --quiet
    """
    p = argparse.ArgumentParser(description="Simple sweep over hidden_dim and lr")
    p.add_argument("--dataset", type=str, default="ESOL")
    p.add_argument("--model", type=str, default="gin")
    p.add_argument("--split", type=str, default="scaffold")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--out-dir", type=str, default="runs/sweep")

    # Small grid
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 128])
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 5e-4])

    # Quality of life / pass-through flags
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--standardize-targets", action="store_true")
    p.add_argument("--scheduler", type=str, default="none")
    p.add_argument("--warmup-epochs", type=int, default=0)

    args = p.parse_args(argv)

    # Build the base CLI argv for the training loop
    base: List[str] = [
        "--dataset", args.dataset,
        "--model", args.model,
        "--split", args.split,
        "--epochs", str(args.epochs),
        "--out-dir", args.out_dir,
    ]
    if args.quiet:
        base.append("--quiet")
    if args.standardize_targets:
        base.append("--standardize-targets")
    if args.scheduler != "none":
        base.extend(["--scheduler", args.scheduler])
        if args.scheduler == "cosine_warmup":
            base.extend(["--warmup-epochs", str(args.warmup_epochs)])

    os.makedirs(args.out_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for hd in args.hidden_dims:
        for lr in args.lrs:
            res = _run_single(base, {"hidden_dim": hd, "lr": lr})
            results.append(res)

    # Pick best by lowest best_val_MAE
    best = min(results, key=lambda r: r.get("best_val_MAE", float("inf")))
    with open(os.path.join(args.out_dir, "sweep_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"best": best, "all": results}, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
