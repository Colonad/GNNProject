# src/datamodules/esol.py
"""
ESOL DataModule (PyTorch Geometric)
----------------------------------
- Auto-downloads ESOL via PyG MoleculeNet.
- Supports random and scaffold splits with seed control.
- Exposes train/val/test DataLoaders and the SMILES list.
- Provides dataset statistics and basic sanity checks.

Definition of Done (Phase 1):
- len(train/val/test) > 0
- batch = next(iter(train_loader)) works without errors
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Literal, Sequence, Tuple, Dict, Any, List, Optional
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


SplitType = Literal["random", "scaffold"]

__all__ = [
    "ESOLConfig",
    "make_dataloaders",
    "load_dataset",
    "scaffold_split",
    "random_split",
    "dataset_stats",
]


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class ESOLConfig:
    root: str = "data"
    batch_size: int = 64
    num_workers: int = 2
    split: SplitType = "scaffold"  # "random" or "scaffold"
    seed: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    drop_last: bool = False
    train_frac: float = 0.8
    val_frac: float = 0.1
    # If set, keep only first N molecules (for very quick dry-runs)
    limit_n: Optional[int] = None
    # Whether to print dataset stats at load time
    verbose: bool = True


# -----------------------------
# Utilities
# -----------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)


def _validate_fracs(train_frac: float, val_frac: float) -> None:
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError(
            f"Invalid fractions: train={train_frac}, val={val_frac} — must be in (0,1) and "
            "train_frac + val_frac < 1."
        )


# -----------------------------
# Splits
# -----------------------------

def scaffold_split(
    smiles_list: Sequence[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Deterministic scaffold split using Bemis–Murcko scaffolds.
    Returns index lists (train, val, test) that are disjoint and cover all items.

    The algorithm sorts scaffold groups by size (desc) to assign large scaffolds first.
    """
    _validate_fracs(train_frac, val_frac)
    N = len(smiles_list)
    n_train = int(math.floor(train_frac * N))
    n_val = int(math.floor(val_frac * N))

    # Group indices by scaffold
    scaff2idx: Dict[str, List[int]] = {}
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            # Treat invalid SMILES as its own group to avoid losing samples
            key = f"INVALID_{i}"
        else:
            key = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaff2idx.setdefault(key, []).append(i)

    # Sort groups by size (desc), tie-broken deterministically by key
    groups = sorted(scaff2idx.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    # Greedy assignment: fill train up to n_train, then val up to n_val, rest → test
    for _, idxs in groups:
        if len(train_idx) + len(idxs) <= n_train:
            train_idx.extend(idxs)
        elif len(val_idx) + len(idxs) <= n_val:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    # If due to rounding we missed a few, push to test
    leftovers = N - (len(train_idx) + len(val_idx) + len(test_idx))
    if leftovers > 0:
        # Deterministic shuffle of remaining indices
        rng = np.random.default_rng(seed)
        remaining = [i for i in range(N) if i not in set(train_idx) | set(val_idx) | set(test_idx)]
        rng.shuffle(remaining)
        test_idx.extend(remaining[:leftovers])

    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == N

    return train_idx, val_idx, test_idx


def random_split(
    N: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """Random permutation split with seed control."""
    _validate_fracs(train_frac, val_frac)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g).tolist()
    n_train = int(math.floor(train_frac * N))
    n_val = int(math.floor(val_frac * N))
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


# -----------------------------
# Dataset loading & stats
# -----------------------------

def load_dataset(cfg: ESOLConfig):
    """
    Load ESOL via PyG MoleculeNet. Returns the PyG dataset object.
    Applies limit_n if provided.
    """
    ds = MoleculeNet(root=cfg.root, name="ESOL")
    if cfg.limit_n is not None:
        # MoleculeNet returns an InMemoryDataset, slicing preserves attributes.
        ds = ds[: int(cfg.limit_n)]
    return ds


@torch.no_grad()
def dataset_stats(ds) -> Dict[str, Any]:
    """Compute basic dataset statistics."""
    y_vals = []
    num_nodes = []
    num_edges = []
    for data in ds:
        # y can be shape [1] per-graph
        y = float(data.y.view(-1)[0].item())
        if not (np.isfinite(y)):
            continue
        y_vals.append(y)
        num_nodes.append(int(data.num_nodes))
        # edge_index has 2 x E
        num_edges.append(int(data.edge_index.size(1)) if data.edge_index is not None else 0)

    stats = {
        "num_graphs": len(ds),
        "y_mean": float(np.mean(y_vals)) if y_vals else float("nan"),
        "y_std": float(np.std(y_vals)) if y_vals else float("nan"),
        "avg_num_nodes": float(np.mean(num_nodes)) if num_nodes else float("nan"),
        "avg_num_edges": float(np.mean(num_edges)) if num_edges else float("nan"),
    }
    return stats


# -----------------------------
# Dataloaders
# -----------------------------

def make_dataloaders(cfg: ESOLConfig):
    """
    Returns:
      (train_loader, val_loader, test_loader, meta)
    where `meta` contains: smiles (list), indices (dict), and dataset stats.
    """
    _set_seed(cfg.seed)

    # Load dataset
    ds = load_dataset(cfg)

    # Extract SMILES strings if available (PyG MoleculeNet stores `smiles`)
    smiles: List[str] = []
    for d in ds:
        s = getattr(d, "smiles", None)
        if s is None:
            # Fallback: reconstruct from RDKit Mol if present (unlikely needed here)
            s = ""
        smiles.append(s)

    # Build splits
    if cfg.split == "scaffold":
        tr_idx, va_idx, te_idx = scaffold_split(
            smiles, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed
        )
    elif cfg.split == "random":
        tr_idx, va_idx, te_idx = random_split(
            len(ds), train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed
        )
    else:
        raise ValueError(f"Unknown split type: {cfg.split!r}")

    # Subset datasets
    ds_train = Subset(ds, tr_idx)
    ds_val = Subset(ds, va_idx)
    ds_test = Subset(ds, te_idx)

    # Reproducible shuffling for train
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=g,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )

    meta = {
        "config": asdict(cfg),
        "indices": {"train": tr_idx, "val": va_idx, "test": te_idx},
        "smiles": smiles,
        "stats": dataset_stats(ds),
    }

    if cfg.verbose:
        s = meta["stats"]
        print(
            f"[ESOL] graphs={s['num_graphs']} | y_mean={s['y_mean']:.3f}±{s['y_std']:.3f} | "
            f"avg_nodes={s['avg_num_nodes']:.1f} | avg_edges={s['avg_num_edges']:.1f}"
        )
        print(
            f"[ESOL] split={cfg.split} | sizes: train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}"
        )

    return train_loader, val_loader, test_loader, meta


# -----------------------------
# Smoke test
# -----------------------------

def _smoke_test() -> None:
    cfg = ESOLConfig(root="data", split="scaffold", batch_size=32, num_workers=0, seed=0, verbose=True)
    train_loader, val_loader, test_loader, meta = make_dataloaders(cfg)
    assert len(train_loader) > 0 and len(val_loader) > 0 and len(test_loader) > 0
    batch = next(iter(train_loader))
    # Expect attributes: x, edge_index, (edge_attr optional), y, batch
    assert hasattr(batch, "x") and hasattr(batch, "edge_index") and hasattr(batch, "y") and hasattr(batch, "batch")
    print("[ESOL] Smoke test passed. First batch:", batch.x.shape, batch.edge_index.shape, batch.y.shape)


if __name__ == "__main__":
    _smoke_test()
