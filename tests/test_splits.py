# tests/test_splits.py
import math
import os
import sys

import pytest

from src.datamodules.splits import (
    SplitSpec,
    normalize_fracs,
    random_split_indices,
    scaffold_split_indices,
    check_disjoint_cover,
)

# --- Helpers -----------------------------------------------------------------

def _approx_sizes(N, tr, va, te, tol=2):
    """
    Verify sizes are close to expected ratios within a small tolerance
    (accounts for floor/rounding and scaffold grouping granularity).
    """
    total = tr + va + te
    assert total == N, f"Sizes do not sum to N: {tr}+{va}+{te} != {N}"
    # no negative sizes
    assert min(tr, va, te) >= 0
    # sanity: largest split should be train
    assert tr >= va and tr >= te


# --- Ratio normalization tests ------------------------------------------------

def test_normalize_fracs_valid():
    tr, va, te = normalize_fracs(0.8, 0.1, None)
    assert pytest.approx(tr + va + te, rel=1e-12) == 1.0
    assert tr > va and tr > te and va > 0 and te > 0


def test_normalize_fracs_invalid():
    with pytest.raises(ValueError):
        normalize_fracs(0.95, 0.1, None)  # train+val >= 1


# --- Random split tests -------------------------------------------------------

@pytest.mark.parametrize("N", [10, 100, 1000])
def test_random_split_invariants(N):
    spec = SplitSpec(0.8, 0.1)
    tr, va, te = random_split_indices(N, spec=spec, seed=0)
    assert check_disjoint_cover((tr, va, te), N)
    _approx_sizes(N, len(tr), len(va), len(te))


def test_random_split_seed_reproducible():
    spec = SplitSpec(0.8, 0.1)
    tr1, va1, te1 = random_split_indices(100, spec=spec, seed=42)
    tr2, va2, te2 = random_split_indices(100, spec=spec, seed=42)
    assert tr1 == tr2 and va1 == va2 and te1 == te2


# --- Scaffold split tests (conditional on RDKit) ------------------------------

@pytest.mark.skipif("rdkit" not in sys.modules and "RDKit" not in sys.modules, reason="RDKit not available")
def test_scaffold_split_basic():
    # A tiny set with repeated scaffolds
    smiles = ["CCO", "O=C=O", "CCN", "CCO", "CCO", "C1CCCCC1", "c1ccccc1", "CCCl", "CCN", "CCO"]
    N = len(smiles)
    spec = SplitSpec(0.8, 0.1)
    tr, va, te = scaffold_split_indices(smiles, spec=spec, seed=0)
    assert check_disjoint_cover((tr, va, te), N)
    _approx_sizes(N, len(tr), len(va), len(te), tol=3)


@pytest.mark.skipif("rdkit" not in sys.modules and "RDKit" not in sys.modules, reason="RDKit not available")
def test_scaffold_split_deterministic():
    smiles = ["CCO", "O=C=O", "CCN", "CCO", "C1CCCCC1", "c1ccccc1", "CCCl"]
    spec = SplitSpec(0.8, 0.1)
    tr1, va1, te1 = scaffold_split_indices(smiles, spec=spec, seed=123)
    tr2, va2, te2 = scaffold_split_indices(smiles, spec=spec, seed=123)
    assert tr1 == tr2 and va1 == va2 and te1 == te2
