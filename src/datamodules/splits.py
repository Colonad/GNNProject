# src/datamodules/splits.py
"""
Split utilities (random + Bemis–Murcko scaffold) with seed control.

This module centralizes split logic used by datamodules:
- Deterministic random split.
- Deterministic scaffold split (groups molecules by core scaffold).
- Ratio validation & normalization helpers.
- Invariant checks (disjoint cover, size expectations).

Definitions
-----------
- Random split: seeded permutation then slice by ratios.
- Scaffold split: compute Bemis–Murcko scaffold for each SMILES, group indices
  by scaffold, sort groups by size (desc), then greedily fill train→val→test.

Notes
-----
- Scaffold split requires RDKit. If RDKit is not available, a clear error is raised.
- Ratios are validated (0<r<1 and sum(train,val)<1). test_frac is inferred if omitted.
- All functions return **index lists** (List[int]).

Example
-------
>>> spec = SplitSpec(train_frac=0.8, val_frac=0.1)
>>> tr, va, te = random_split_indices(N=100, spec=spec, seed=0)
>>> assert check_disjoint_cover((tr,va,te), N=100)

>>> smiles = ["CCO", "O=C=O", "CCN", ...]
>>> tr, va, te = scaffold_split_indices(smiles, spec=spec, seed=0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import math
import numpy as np
import torch

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    _HAS_RDKIT = True
except Exception:  # pragma: no cover
    Chem = None
    MurckoScaffold = None
    _HAS_RDKIT = False


__all__ = [
    "SplitSpec",
    "normalize_fracs",
    "random_split_indices",
    "scaffold_groups",
    "scaffold_split_indices",
    "check_disjoint_cover",
]


# -----------------------------------------------------------------------------
# Spec & ratio helpers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitSpec:
    """Split ratios (train/val[/test]). If test_frac is None, it's inferred as 1-train-val."""
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: Optional[float] = None

    def normalized(self) -> "SplitSpec":
        tr, va, te = normalize_fracs(self.train_frac, self.val_frac, self.test_frac)
        return SplitSpec(tr, va, te)

    def sizes(self, N: int) -> Tuple[int, int, int]:
        tr, va, te = self.normalized().as_tuple()
        n_train = int(math.floor(tr * N))
        n_val = int(math.floor(va * N))
        n_test = max(0, N - (n_train + n_val))
        return n_train, n_val, n_test

    def as_tuple(self) -> Tuple[float, float, float]:
        spec = self.normalized()
        return spec.train_frac, spec.val_frac, spec.test_frac or 0.0


def _validate_fracs(train_frac: float, val_frac: float) -> None:
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and (train_frac + val_frac) < 1):
        raise ValueError(
            f"Invalid fractions: train={train_frac}, val={val_frac}. "
            "They must be in (0,1) and train+val < 1."
        )


def normalize_fracs(train_frac: float, val_frac: float, test_frac: Optional[float] = None) -> Tuple[float, float, float]:
    """Validate ratios; infer test if missing; return (train, val, test)."""
    _validate_fracs(train_frac, val_frac)
    if test_frac is None:
        test_frac = 1.0 - (train_frac + val_frac)
    total = train_frac + val_frac + test_frac
    if total <= 0:
        raise ValueError("Sum of fractions must be positive.")
    # Normalize to sum to 1 (handles tiny floating error)
    tr, va, te = train_frac / total, val_frac / total, test_frac / total
    return tr, va, te


# -----------------------------------------------------------------------------
# Random split
# -----------------------------------------------------------------------------

def random_split_indices(
    N: int,
    spec: SplitSpec = SplitSpec(),
    seed: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """Seeded random split returning (train_idx, val_idx, test_idx)."""
    tr, va, te = spec.normalized().as_tuple()
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(N, generator=g).tolist()
    n_train = int(math.floor(tr * N))
    n_val = int(math.floor(va * N))
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


# -----------------------------------------------------------------------------
# Scaffold split
# -----------------------------------------------------------------------------

def scaffold_groups(smiles_list: Sequence[str]) -> Dict[str, List[int]]:
    """
    Group indices by Bemis–Murcko scaffold string.
    If an SMILES is invalid or RDKit cannot parse, it forms its own group.
    """
    if not _HAS_RDKIT:
        raise ImportError("RDKit is required for scaffold grouping but is not available.")
    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s) if s is not None else None
        key = f"INVALID_{i}" if (mol is None) else MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        groups.setdefault(key, []).append(i)
    return groups


def scaffold_split_indices(
    smiles_list: Sequence[str],
    spec: SplitSpec = SplitSpec(),
    seed: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Deterministic scaffold split:
    1) Group by scaffold.
    2) Sort groups by size (desc) then by scaffold string (asc) for tie-breaking.
    3) Greedily fill train up to n_train, then val up to n_val, remainder → test.
    """
    if not _HAS_RDKIT:
        raise ImportError("RDKit is required for scaffold split but is not available.")
    tr, va, te = spec.normalized().as_tuple()
    N = len(smiles_list)
    n_train = int(math.floor(tr * N))
    n_val = int(math.floor(va * N))

    groups = sorted(scaffold_groups(smiles_list).items(), key=lambda kv: (-len(kv[1]), kv[0]))
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

    # Any leftovers due to rounding go to test (deterministically shuffled)
    leftovers = N - (len(train_idx) + len(val_idx) + len(test_idx))
    if leftovers > 0:
        rng = np.random.default_rng(int(seed))
        remaining = [i for i in range(N) if i not in set(train_idx) | set(val_idx) | set(test_idx)]
        rng.shuffle(remaining)
        test_idx.extend(remaining[:leftovers])

    assert check_disjoint_cover((train_idx, val_idx, test_idx), N), "Split indices do not form a disjoint cover."
    return train_idx, val_idx, test_idx


# -----------------------------------------------------------------------------
# Invariants / checks
# -----------------------------------------------------------------------------

def check_disjoint_cover(idxs: Tuple[Sequence[int], Sequence[int], Sequence[int]], N: int) -> bool:
    """Return True iff the three index sets are pairwise disjoint and cover 0..N-1 exactly."""
    a, b, c = (set(map(int, idxs[0])), set(map(int, idxs[1])), set(map(int, idxs[2])))
    disjoint = (a & b == set()) and (a & c == set()) and (b & c == set())
    cover = (len(a | b | c) == N)
    in_range = all(0 <= i < N for i in (a | b | c))
    return bool(disjoint and cover and in_range)


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

def _smoke_test() -> None:  # pragma: no cover
    N = 100
    spec = SplitSpec(0.8, 0.1)
    r_tr, r_va, r_te = random_split_indices(N, spec=spec, seed=0)
    assert check_disjoint_cover((r_tr, r_va, r_te), N)
    print("[splits] random OK:", len(r_tr), len(r_va), len(r_te))

    # Scaffold test with a few toy SMILES
    smiles = ["CCO", "O=C=O", "CCN", "CCO", "CCO", "C1CCCCC1", "c1ccccc1", "CCCl"]
    if _HAS_RDKIT:
        s_tr, s_va, s_te = scaffold_split_indices(smiles, spec=spec, seed=0)
        assert check_disjoint_cover((s_tr, s_va, s_te), len(smiles))
        print("[splits] scaffold OK:", len(s_tr), len(s_va), len(s_te))
    else:
        print("[splits] RDKit not available; scaffold test skipped.")


if __name__ == "__main__":  # pragma: no cover
    _smoke_test()
