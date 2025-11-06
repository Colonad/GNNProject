# src/datamodules/qm9.py
"""
QM9 DataModule (PyTorch Geometric)
----------------------------------
- Auto-downloads QM9 via PyG.
- Subsamples the first N molecules (default: 15k) for laptop-friendly runs.
- Exposes a SINGLE scalar target (e.g., U0) by selecting the appropriate column of y.
- Provides random/scaffold splits with seed control and summary stats.
- Adds basic node features `x` from atomic numbers `z` if missing (H,C,N,O,F one-hot).

Definition of Done (Phase 1):
- len(train/val/test) > 0
- batch = next(iter(train_loader)) works without errors
- prints dataset stats to log
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Any

import math
import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.datasets import QM9 as PYG_QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform, Compose

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# Import scaffold split from ESOL module for consistency
try:
    from .esol import scaffold_split as esol_scaffold_split
except Exception:
    esol_scaffold_split = None  # Fallback: we'll define a local version if needed


SplitType = Literal["random", "scaffold"]

# Common 12-target naming for QM9 (PyG style)
# Index mapping widely used in literature/examples:
# 0: mu, 1: alpha, 2: homo, 3: lumo, 4: gap, 5: r2, 6: zpve,
# 7: U0, 8: U, 9: H, 10: G, 11: Cv
QM9_TARGET_INDEX: Dict[str, int] = {
    "mu": 0,
    "alpha": 1,
    "homo": 2,
    "lumo": 3,
    "gap": 4,
    "r2": 5,
    "zpve": 6,
    "U0": 7,
    "U": 8,
    "H": 9,
    "G": 10,
    "Cv": 11,
}

__all__ = [
    "QM9Config",
    "make_dataloaders",
    "load_dataset",
    "dataset_stats",
    "random_split",
]


@dataclass
class QM9Config:
    root: str = "data"
    batch_size: int = 64
    num_workers: int = 2
    split: SplitType = "scaffold"  # or "random"
    seed: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    drop_last: bool = False
    train_frac: float = 0.8
    val_frac: float = 0.1
    # Subsample size for laptop runs (QM9 has ~134k molecules)
    limit_n: Optional[int] = 15000
    # Select target by name (e.g., "U0") or by index if None/invalid
    target_key: Optional[str] = "U0"
    target_index: Optional[int] = None  # overrides key if provided
    # Whether to print dataset stats at load time
    verbose: bool = True


# -----------------------------
# Utilities
# -----------------------------

def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)


def _validate_fracs(train_frac: float, val_frac: float) -> None:
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError(
            f"Invalid fractions: train={train_frac}, val={val_frac} — must be in (0,1) and "
            "train_frac + val_frac < 1."
        )


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


def _local_scaffold_split(smiles_list: Sequence[str], train_frac: float, val_frac: float, seed: int):
    """Local scaffold split if ESOL's function is not importable (identical logic)."""
    _validate_fracs(train_frac, val_frac)
    N = len(smiles_list)
    n_train = int(math.floor(train_frac * N))
    n_val = int(math.floor(val_frac * N))

    scaff2idx: Dict[str, List[int]] = {}
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        key = f"INVALID_{i}" if mol is None else MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaff2idx.setdefault(key, []).append(i)

    groups = sorted(scaff2idx.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for _, idxs in groups:
        if len(train_idx) + len(idxs) <= n_train:
            train_idx.extend(idxs)
        elif len(val_idx) + len(idxs) <= n_val:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    leftovers = N - (len(train_idx) + len(val_idx) + len(test_idx))
    if leftovers > 0:
        rng = np.random.default_rng(seed)
        remaining = [i for i in range(N) if i not in set(train_idx) | set(val_idx) | set(test_idx)]
        rng.shuffle(remaining)
        test_idx.extend(remaining[:leftovers])

    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == N

    return train_idx, val_idx, test_idx


from torch_geometric.transforms import BaseTransform, Compose

class EnsureNodeFeatures(BaseTransform):
    """
    If `data.x` is missing, build a basic node feature matrix from atomic numbers `z`:
      - one-hot over {H,C,N,O,F} ordered as [1,6,7,8,9].
    Leaves existing data.x untouched.
    """
    def __init__(self):
        super().__init__()
        self.allowed = [1, 6, 7, 8, 9]  # H, C, N, O, F

    def __call__(self, data):
        if getattr(data, "x", None) is None or data.x is None:
            z = getattr(data, "z", None)
            if z is None:
                return data  # nothing to do
            one_hot = torch.zeros((z.numel(), len(self.allowed)), dtype=torch.float)
            for i, zi in enumerate(z.view(-1).tolist()):
                if zi in self.allowed:
                    one_hot[i, self.allowed.index(zi)] = 1.0
            data.x = one_hot
        return data


class SelectQM9Target(BaseTransform):
    """
    Selects a single target column from QM9.y and reshapes it to 1D (per-graph scalar).
    """
    def __init__(self, index: int):
        super().__init__()
        if not isinstance(index, int) or index < 0:
            raise ValueError(f"Invalid target index: {index!r}")
        self.index = index

    def __call__(self, data):
        y = getattr(data, "y", None)
        if y is None:
            return data
        y = y.view(-1)
        if self.index >= y.numel():
            raise IndexError(f"Target index {self.index} out of bounds for y with {y.numel()} dims")
        data.y = y[self.index].view(1)
        return data


# -----------------------------
# Dataset loading & stats
# -----------------------------

def _resolve_target_index(target_key: Optional[str], target_index: Optional[int]) -> int:
    if target_index is not None:
        return int(target_index)
    if target_key is not None and target_key in QM9_TARGET_INDEX:
        return QM9_TARGET_INDEX[target_key]
    return QM9_TARGET_INDEX["U0"]  # default


def load_dataset(cfg: QM9Config):
    """
    Load QM9 via PyG. Applies transforms to ensure x is present and y is a single column.
    Subsamples to limit_n if configured.
    """
    target_idx = _resolve_target_index(cfg.target_key, cfg.target_index)
    transform = Compose([EnsureNodeFeatures(), SelectQM9Target(target_idx)])
    ds = PYG_QM9(root=cfg.root, transform=transform)

    if cfg.limit_n is not None:
        ds = ds[: int(cfg.limit_n)]
    return ds


@torch.no_grad()
def dataset_stats(ds) -> Dict[str, Any]:
    """Compute basic dataset statistics for the selected target (y)."""
    y_vals = []
    num_nodes = []
    num_edges = []
    for data in ds:
        y = float(data.y.view(-1)[0].item())
        if not (np.isfinite(y)):
            continue
        y_vals.append(y)
        num_nodes.append(int(data.num_nodes))
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

def make_dataloaders(cfg: QM9Config):
    """
    Returns:
      (train_loader, val_loader, test_loader, meta)
    where `meta` contains: smiles (list), indices (dict), and dataset stats.
    """
    _set_seed(cfg.seed)

    # Load dataset with transforms (x present, y scalar)
    ds = load_dataset(cfg)

    # Extract SMILES
    smiles: List[str] = [getattr(d, "smiles", "") for d in ds]

    # Build splits
    if cfg.split == "scaffold":
        if esol_scaffold_split is not None:
            tr_idx, va_idx, te_idx = esol_scaffold_split(
                smiles, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed
            )
        else:
            tr_idx, va_idx, te_idx = _local_scaffold_split(
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
    g = torch.Generator().manual_seed(cfg.seed)

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
        tgt = cfg.target_key if cfg.target_index is None else f"idx{cfg.target_index}"
        print(
            f"[QM9] target={tgt} | graphs={s['num_graphs']} | y_mean={s['y_mean']:.4f}±{s['y_std']:.4f} | "
            f"avg_nodes={s['avg_num_nodes']:.1f} | avg_edges={s['avg_num_edges']:.1f}"
        )
        print(
            f"[QM9] split={cfg.split} | sizes: train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}"
        )

    return train_loader, val_loader, test_loader, meta


# -----------------------------
# Smoke test
# -----------------------------

def _smoke_test() -> None:
    cfg = QM9Config(root="data", split="scaffold", batch_size=32, num_workers=0, seed=0, verbose=True)
    train_loader, val_loader, test_loader, meta = make_dataloaders(cfg)
    assert len(train_loader) > 0 and len(val_loader) > 0 and len(test_loader) > 0
    batch = next(iter(train_loader))
    assert hasattr(batch, "x") and hasattr(batch, "edge_index") and hasattr(batch, "y") and hasattr(batch, "batch")
    # y should be scalar per graph
    assert batch.y.dim() == 1 or (batch.y.dim() == 2 and batch.y.size(-1) == 1)
    print("[QM9] Smoke test passed. First batch:", batch.x.shape, batch.edge_index.shape, batch.y.shape)


if __name__ == "__main__":
    _smoke_test()
