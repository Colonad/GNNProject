#!/usr/bin/env python
# src/interpret/substructure_mask.py
from __future__ import annotations


"""
Substructure masking for ESOL (Phase 7)
======================================

This script:
  - Loads a trained GIN / MPNN checkpoint from a Phase-6 run directory.
  - Loads ESOL from torch_geometric.datasets.MoleculeNet.
  - Picks a small set of molecules (default: 8).
  - For each molecule:
      * Compute baseline prediction.
      * Mask:
          - ring atoms (IsInRing)
          - hetero atoms (non-carbon heavy atoms: Z != 6 and != 1)
        by zeroing their node features.
      * Recompute prediction.
      * Log Δprediction = masked_pred - baseline_pred.

Outputs
-------
  - CSV: report/tables/substructure_mask_{dataset}_{model}.csv
  - Figure: report/figures/substructure_mask_{dataset}_{model}.png

Usage
-----
  python -m src.interpret.substructure_mask \
    --run-dir runs/esol_gin_scaffold_seed0_20251128-164102 \
    --model gin \
    --dataset ESOL \
    --num-mols 8
"""

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import Batch

import numpy as np
import matplotlib.pyplot as plt

try:
    from rdkit import Chem
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "rdkit is required for substructure masking. "
        "Install via conda (recommended): `conda install -c rdkit rdkit`."
    ) from e

from src.models.gin import GINConfig, GINNet
from src.models.mpnn import MPNNConfig, MPNNNet


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

@dataclass
class MaskSpec:
    name: str
    indices: List[int]


def _infer_device(device_str: str | None = None) -> torch.device:
    if device_str is not None and device_str.lower() != "auto":
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    # Try common keys; fall back to treating the whole object as state_dict.
    if isinstance(ckpt, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                model.load_state_dict(ckpt[key], strict=False)
                return
    if isinstance(ckpt, dict):
        model.load_state_dict(ckpt, strict=False)
    else:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} has unexpected format: {type(ckpt)}"
        )


def _build_dataset(dataset: str, root: str) -> MoleculeNet:
    return MoleculeNet(root=root, name=dataset)


def _build_model(
    model_name: str,
    dataset: MoleculeNet,
    hidden_dim: int = 128,
    num_layers: int = 6,
) -> nn.Module:
    model_name = model_name.lower()
    in_dim = dataset.num_node_features
    out_dim = 1

    if model_name == "gin":
        cfg = GINConfig(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        return GINNet(cfg)
    elif model_name == "mpnn":
        # Edge attributes required
        if getattr(dataset[0], "edge_attr", None) is None:
            raise ValueError(
                "MPNN requires edge_attr in the dataset, but edge_attr is None."
            )
        edge_dim = int(dataset[0].edge_attr.size(-1))
        cfg = MPNNConfig(
            in_dim=in_dim,
            edge_dim=edge_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        return MPNNNet(cfg)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def _pick_indices(num_total: int, num_mols: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    num_mols = min(num_mols, num_total)
    return sorted(rng.sample(range(num_total), k=num_mols))


def _compute_masks_for_mol(smiles: str, num_nodes: int) -> List[MaskSpec]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    ring_idx: List[int] = []
    hetero_idx: List[int] = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx >= num_nodes:
            continue
        if atom.IsInRing():
            ring_idx.append(idx)
        z = atom.GetAtomicNum()
        if z not in (1, 6):  # non-H, non-C
            hetero_idx.append(idx)

    specs: List[MaskSpec] = []
    if ring_idx:
        specs.append(MaskSpec(name="ring_atoms", indices=sorted(set(ring_idx))))
    if hetero_idx:
        specs.append(MaskSpec(name="hetero_atoms", indices=sorted(set(hetero_idx))))
    return specs


def _predict_single(model: nn.Module, data, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        batch = Batch.from_data_list([data]).to(device)
        out = model(batch)
        # Expect shape [1, 1] or [1]; flatten to scalar.
        if isinstance(out, torch.Tensor):
            return float(out.view(-1)[0].item())
        raise RuntimeError(f"Unexpected model output type: {type(out)}")


# ---------------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------------

def run_substructure_masking(
    run_dir: str,
    model_name: str,
    dataset_name: str = "ESOL",
    data_root: str = "data",
    ckpt_name: str = "best.ckpt",
    num_mols: int = 8,
    seed: int = 0,
    out_csv: str | None = None,
    out_fig: str | None = None,
    device_str: str | None = None,
) -> Tuple[str, str]:
    device = _infer_device(device_str)
    run_dir = os.path.abspath(run_dir)
    ckpt_path = os.path.join(run_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    dataset = _build_dataset(dataset_name, data_root)
    model = _build_model(model_name, dataset)
    model.to(device)
    _load_checkpoint(model, ckpt_path, device)

    # Default outputs
    tables_dir = os.path.join("report", "tables")
    figs_dir = os.path.join("report", "figures")
    _ensure_dir(tables_dir)
    _ensure_dir(figs_dir)

    if out_csv is None:
        out_csv = os.path.join(
            tables_dir, f"substructure_mask_{dataset_name}_{model_name}.csv"
        )
    if out_fig is None:
        out_fig = os.path.join(
            figs_dir, f"substructure_mask_{dataset_name}_{model_name}.png"
        )

    # Pick which molecules to probe
    idxs = _pick_indices(len(dataset), num_mols, seed)

    rows: List[Dict[str, Any]] = []

    for idx in idxs:
        data = dataset[idx]
        num_nodes = int(data.num_nodes)
        smiles = getattr(data, "smiles", None)
        if smiles is None:
            # MoleculeNet typically stores 'smiles'; if not, skip.
            continue

        masks = _compute_masks_for_mol(smiles, num_nodes)
        if not masks:
            # If no rings / hetero atoms, still log baseline.
            masks = []

        # Baseline
        base_pred = _predict_single(model, data, device)
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "run_dir": run_dir,
                "idx": idx,
                "smiles": smiles,
                "mask_type": "baseline",
                "baseline_pred": base_pred,
                "masked_pred": base_pred,
                "delta": 0.0,
            }
        )

        # Mask each spec
        for spec in masks:
            masked = data.clone()
            masked.x = masked.x.clone()
            if spec.indices:
                idx_tensor = torch.tensor(spec.indices, dtype=torch.long)
                masked.x[idx_tensor] = 0.0

            masked_pred = _predict_single(model, masked, device)
            delta = masked_pred - base_pred

            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "run_dir": run_dir,
                    "idx": idx,
                    "smiles": smiles,
                    "mask_type": spec.name,
                    "baseline_pred": base_pred,
                    "masked_pred": masked_pred,
                    "delta": delta,
                }
            )

    # Write CSV
    fieldnames = [
        "dataset",
        "model",
        "run_dir",
        "idx",
        "smiles",
        "mask_type",
        "baseline_pred",
        "masked_pred",
        "delta",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Build figure: Δprediction by mask type (masked - baseline)
    deltas_by_type: Dict[str, List[float]] = {}
    for r in rows:
        mt = r["mask_type"]
        if mt == "baseline":
            continue
        deltas_by_type.setdefault(mt, []).append(float(r["delta"]))

    if deltas_by_type:
        mask_types = sorted(deltas_by_type.keys())
        x = np.arange(len(mask_types))

        plt.figure(figsize=(6, 4))
        # Scatter individual molecule deltas + mean bar
        for j, mt in enumerate(mask_types):
            vals = np.asarray(deltas_by_type[mt], dtype=float)
            # Jittered scatter
            jitter = (np.random.rand(len(vals)) - 0.5) * 0.15
            plt.scatter(
                x[j] + jitter,
                vals,
                alpha=0.7,
                label=f"{mt} (N={len(vals)})" if j == 0 else None,
            )
            mean = float(vals.mean()) if len(vals) > 0 else 0.0
            plt.bar(
                x[j],
                mean,
                width=0.4,
                alpha=0.3,
                edgecolor="black",
            )

        plt.axhline(0.0, linestyle="--")
        plt.xticks(x, mask_types, rotation=30)
        plt.ylabel("Δprediction (masked - baseline)")
        plt.title(f"ESOL substructure masking — {model_name.upper()}")
        plt.tight_layout()
        plt.savefig(out_fig, dpi=200)
        plt.close()

    return out_csv, out_fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Substructure masking: mask rings/heteroatoms and measure Δprediction."
    )
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory containing best.ckpt (phase-6 run).",
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gin", "mpnn"],
        help="Model type used in the run.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="ESOL",
        help="MoleculeNet dataset name (default: ESOL).",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory for MoleculeNet datasets (default: data).",
    )
    p.add_argument(
        "--ckpt-name",
        type=str,
        default="best.ckpt",
        help="Checkpoint filename inside run-dir (default: best.ckpt).",
    )
    p.add_argument(
        "--num-mols",
        type=int,
        default=8,
        help="Number of molecules to probe (default: 8).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for molecule selection.",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional output CSV path (default: report/tables/substructure_mask_*.csv).",
    )
    p.add_argument(
        "--out-fig",
        type=str,
        default=None,
        help="Optional output figure path (default: report/figures/substructure_mask_*.png).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device string (e.g., 'cuda', 'cpu', 'auto').",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    csv_path, fig_path = run_substructure_masking(
        run_dir=args.run_dir,
        model_name=args.model,
        dataset_name=args.dataset,
        data_root=args.data_root,
        ckpt_name=args.ckpt_name,
        num_mols=args.num_mols,
        seed=args.seed,
        out_csv=args.out_csv,
        out_fig=args.out_fig,
        device_str=args.device,
    )
    print(f"[substructure_mask] Wrote CSV  -> {csv_path}")
    print(f"[substructure_mask] Wrote fig  -> {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
