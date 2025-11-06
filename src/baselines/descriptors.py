# src/baselines/descriptors.py
"""
Descriptor Baselines (RDKit) for Molecular Property Prediction
==============================================================

- Robust SMILES parsing and descriptor computation via RDKit.
- Optional Morgan fingerprints using the modern rdFingerprintGenerator.MorganGenerator
  (falls back to legacy API if unavailable), so no deprecation warnings.
- Sanitization, constant-column drop, optional scaling, and NPZ persistence.
- CLI to build features for ESOL or a QM9 subset.

Definition of Done:
- Produce (N, D) feature matrix with finite entries only; save to outputs/descriptors/*.npz

Backward-compat:
- Retains `build_featurizer(...)` + `DescriptorConfig` (featurizer settings) used by src/cli/baseline.py.
- Adds a separate `DescriptorMatrixConfig` for the CLI/run-and-save pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Optional sklearn for scaling
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
except Exception:  # pragma: no cover
    StandardScaler = None
    MinMaxScaler = None

# RDKit imports
from rdkit import Chem, RDLogger
from rdkit import DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Silence non-critical RDKit logs
RDLogger.DisableLog("rdApp.debug")
RDLogger.DisableLog("rdApp.info")

# Optional: PyG datasets for convenience in the CLI
try:
    from torch_geometric.datasets import MoleculeNet
    from torch_geometric.datasets import QM9 as PYG_QM9
except Exception:  # pragma: no cover
    MoleculeNet = None
    PYG_QM9 = None


# ======================================================================================
# PART A — Featurizer API (used by src/cli/baseline.py)
# ======================================================================================

@dataclass
class DescriptorConfig:
    """Featurizer configuration (used by classical baselines)."""
    kind: str = "morgan"              # "morgan" | "physchem" | "morgan_physchem"
    # Morgan FP params
    morgan_bits: int = 2048
    morgan_radius: int = 2
    use_chirality: bool = True
    use_features: bool = False
    # Physchem set
    physchem_set: str = "basic"       # currently only "basic"
    # Parallelism (physchem only; joblib optional)
    n_jobs: int = 0


def _smiles_to_mol(smi: str) -> Chem.Mol:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    Chem.SanitizeMol(m)
    return m


def _physchem_feature_names(set_name: str = "basic") -> List[str]:
    if set_name != "basic":
        raise ValueError(f"Unknown physchem_set '{set_name}'. Only 'basic' is supported.")
    # Curated set: fast, stable, informative for small molecules
    return [
        "MolWt",
        "HeavyAtomCount",
        "NumValenceElectrons",
        "FractionCSP3",
        "TPSA",
        "MolLogP",
        "NumHAcceptors",
        "NumHDonors",
        "NumRotatableBonds",
        "RingCount",
        "NumAromaticRings",
        "NumAliphaticRings",
        "NumSaturatedRings",
        "NumHeteroatoms",
        "HallKierAlpha",
        "LabuteASA",
    ]


def _compute_physchem_vector(mol: Chem.Mol, names: Sequence[str]) -> np.ndarray:
    v: List[float] = []
    for name in names:
        if name == "MolWt":
            v.append(float(Descriptors.MolWt(mol)))
        elif name == "HeavyAtomCount":
            v.append(float(Descriptors.HeavyAtomCount(mol)))
        elif name == "NumValenceElectrons":
            v.append(float(Descriptors.NumValenceElectrons(mol)))
        elif name == "FractionCSP3":
            v.append(float(Descriptors.FractionCSP3(mol)))
        elif name == "TPSA":
            v.append(float(rdMolDescriptors.CalcTPSA(mol)))
        elif name == "MolLogP":
            v.append(float(Descriptors.MolLogP(mol)))
        elif name == "NumHAcceptors":
            v.append(float(rdMolDescriptors.CalcNumHBA(mol)))
        elif name == "NumHDonors":
            v.append(float(rdMolDescriptors.CalcNumHBD(mol)))
        elif name == "NumRotatableBonds":
            v.append(float(rdMolDescriptors.CalcNumRotatableBonds(mol)))
        elif name == "RingCount":
            v.append(float(rdMolDescriptors.CalcNumRings(mol)))
        elif name == "NumAromaticRings":
            v.append(float(rdMolDescriptors.CalcNumAromaticRings(mol)))
        elif name == "NumAliphaticRings":
            v.append(float(rdMolDescriptors.CalcNumAliphaticRings(mol)))
        elif name == "NumSaturatedRings":
            v.append(float(rdMolDescriptors.CalcNumSaturatedRings(mol)))
        elif name == "NumHeteroatoms":
            v.append(float(rdMolDescriptors.CalcNumHeteroatoms(mol)))
        elif name == "HallKierAlpha":
            v.append(float(Descriptors.HallKierAlpha(mol)))
        elif name == "LabuteASA":
            v.append(float(rdMolDescriptors.CalcLabuteASA(mol)))
        else:
            raise KeyError(f"Unknown physchem feature: {name}")
    return np.asarray(v, dtype=np.float32)


def _featurize_physchem(smiles_list: Sequence[str], set_name: str = "basic", n_jobs: int = 0) -> np.ndarray:
    names = _physchem_feature_names(set_name)
    n = len(smiles_list)

    def _one(smi: str) -> np.ndarray:
        mol = _smiles_to_mol(smi)
        return _compute_physchem_vector(mol, names)

    # joblib optional / inline by default
    try:
        import joblib  # type: ignore
    except Exception:
        joblib = None  # noqa: F841

    if n_jobs and 'joblib' in globals() and globals()['joblib'] is not None and int(n_jobs) not in (0, 1):
        feats = globals()['joblib'].Parallel(n_jobs=int(n_jobs), prefer="threads")(
            globals()['joblib'].delayed(_one)(s) for s in smiles_list
        )
        X = np.vstack(feats).astype(np.float32, copy=False)
    else:
        X = np.zeros((n, len(names)), dtype=np.float32)
        for i, s in enumerate(smiles_list):
            X[i, :] = _one(s)
    return X


# --- in src/baselines/descriptors.py ---

def _morgan_generator(bits: int, radius: int, use_chirality: bool, use_features: bool):
    """
    Return an RDKit Morgan fingerprint generator compatible across RDKit versions.
    We pass only widely supported kwargs and avoid deprecated/renamed ones.
    """
    try:
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        # Minimal, version-safe signature:
        # GetMorganGenerator(radius=3, countSimulation=False, includeChirality=False,
        #                    useBondTypes=True, onlyNonzeroInvariants=False,
        #                    includeRingMembership=True, fpSize=2048, ...)
        return GetMorganGenerator(
            radius=int(radius),
            countSimulation=False,
            includeChirality=bool(use_chirality),
            useBondTypes=True,
            onlyNonzeroInvariants=False,
            includeRingMembership=True,
            fpSize=int(bits),
        )
    except Exception:
        # Fallback for older RDKit builds: use legacy AllChem bit vectors
        return None  # signal to caller to use legacy path


def _featurize_morgan(smiles: Sequence[str],
                      radius: int,
                      bits: int,
                      use_chirality: bool,
                      use_features: bool) -> np.ndarray:
    N = len(smiles)
    X = np.zeros((N, bits), dtype=np.float32)

    gen = _morgan_generator(bits, radius, use_chirality, use_features)
    if gen is not None:
        # New-style generator path
        from rdkit import Chem
        from rdkit import DataStructs
        for i, smi in enumerate(smiles):
            mol = smiles_to_mol(smi)
            if mol is None:
                continue
            fp = gen.GetFingerprint(mol)
            arr = np.zeros((bits,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            X[i, :] = arr
        return X

    # Legacy path (older RDKit)
    from rdkit.Chem import AllChem as _AllChem
    from rdkit import Chem, DataStructs
    for i, smi in enumerate(smiles):
        mol = smiles_to_mol(smi)
        if mol is None:
            continue
        fp = _AllChem.GetMorganFingerprintAsBitVect(
            mol, int(radius), nBits=int(bits), useChirality=bool(use_chirality)
        )
        arr = np.zeros((bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i, :] = arr
    return X


def build_featurizer(
    *,
    kind: str = "morgan",
    morgan_bits: int = 2048,
    morgan_radius: int = 2,
    use_chirality: bool = True,
    use_features: bool = False,
    physchem_set: str = "basic",
    n_jobs: int = 0,
) -> Callable[[Sequence[str]], np.ndarray]:
    """
    Return a callable: featurizer(smiles_list) -> np.ndarray[n_samples, n_features] (float32).

    kind: "morgan" | "physchem" | "morgan_physchem"
    """
    k = kind.lower()
    if k not in {"morgan", "physchem", "morgan_physchem"}:
        raise ValueError(f"Unknown feature kind '{kind}'")

    def _morgan_only(smiles_list: Sequence[str]) -> np.ndarray:
        return _featurize_morgan(
            smiles_list,
            bits=int(morgan_bits),
            radius=int(morgan_radius),
            use_chirality=bool(use_chirality),
            use_features=bool(use_features),
        )

    def _phys_only(smiles_list: Sequence[str]) -> np.ndarray:
        return _featurize_physchem(smiles_list, set_name=physchem_set, n_jobs=int(n_jobs))

    if k == "morgan":
        return _morgan_only
    if k == "physchem":
        return _phys_only

    def _combo(smiles_list: Sequence[str]) -> np.ndarray:
        X1 = _morgan_only(smiles_list)
        X2 = _phys_only(smiles_list)
        return np.concatenate([X1, X2], axis=1, dtype=np.float32)

    return _combo


# ======================================================================================
# PART B — Full Descriptor Matrix Pipeline + CLI (your requested addition)
# ======================================================================================

@dataclass
class DescriptorMatrixConfig:
    """Config for building and saving a full descriptor matrix from a dataset."""
    dataset: str = "ESOL"                 # ESOL | QM9
    root: str = "data"
    limit_n: Optional[int] = 15000        # QM9 only

    use_morgan: bool = True
    morgan_radius: int = 2
    morgan_bits: int = 1024
    morgan_use_chirality: bool = True

    drop_constant: bool = True
    scale: str = "none"                   # {"none","standard","minmax"}

    out_dir: str = "outputs/descriptors"
    file_prefix: Optional[str] = None
    seed: int = 0
    verbose: bool = True


def get_rdkit_descriptor_names() -> List[str]:
    return [name for (name, _fn) in Descriptors.descList]


def smiles_to_mol(smi: str):
    if smi is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def compute_rdkit_descriptors(smiles_list: Sequence[str]) -> Tuple[np.ndarray, List[str], List[bool]]:
    names = get_rdkit_descriptor_names()
    from rdkit.ML.Descriptors import MoleculeDescriptors  # lazy import to keep top clean
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)

    N, D = len(smiles_list), len(names)
    X = np.zeros((N, D), dtype=float)
    valid = [False] * N

    for i, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        if mol is None:
            continue
        try:
            vals = calc.CalcDescriptors(mol)
        except Exception:
            vals = [float("nan")] * D
        X[i, :] = np.asarray(vals, dtype=float)
        valid[i] = True

    return X, names, valid


def compute_morgan_fingerprints(
    smiles_list: Sequence[str],
    radius: int = 2,
    n_bits: int = 1024,
    use_chirality: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """Morgan FPs via modern generator; falls back to legacy API if needed."""
    N = len(smiles_list)
    X = np.zeros((N, n_bits), dtype=float)
    colnames = [f"morgan_{radius}_bit_{i}" for i in range(n_bits)]

    try:
        gen = GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
            includeChirality=use_chirality,
            useBondTypes=True,
            useCountSimulation=False,
        )
        for i, smi in enumerate(smiles_list):
            mol = smiles_to_mol(smi)
            if mol is None:
                continue
            fp = gen.GetFingerprint(mol)
            arr = np.zeros((n_bits,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            X[i, :] = arr.astype(float)
        return X, colnames
    except Exception:
        # Fallback (older RDKit)
        from rdkit.Chem import AllChem as _AllChem  # type: ignore
        for i, smi in enumerate(smiles_list):
            mol = smiles_to_mol(smi)
            if mol is None:
                continue
            fp = _AllChem.GetMorganFingerprintAsBitVect(
                mol, radius, nBits=n_bits, useChirality=use_chirality
            )
            arr = np.zeros((n_bits,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            X[i, :] = arr.astype(float)
        return X, colnames


def sanitize_matrix(X: np.ndarray, *, replace_nan_inf_with: float = 0.0) -> np.ndarray:
    return np.nan_to_num(X, nan=replace_nan_inf_with, posinf=replace_nan_inf_with, neginf=replace_nan_inf_with)


def drop_constant_columns(X: np.ndarray, colnames: List[str]) -> Tuple[np.ndarray, List[str], np.ndarray]:
    if X.size == 0:
        return X, colnames, np.array([], dtype=bool)
    var = X.var(axis=0)
    keep = var > 0.0
    return X[:, keep], [c for c, k in zip(colnames, keep) if k], keep


def scale_matrix(X: np.ndarray, method: str = "none") -> Tuple[np.ndarray, Dict[str, Any]]:
    method = (method or "none").lower()
    if method == "none":
        return X, {"method": "none"}
    if method == "standard":
        if StandardScaler is None:
            raise ImportError("scikit-learn is required for standard scaling.")
        scaler = StandardScaler().fit(X)
        return scaler.transform(X), {"method": "standard", "mean_": scaler.mean_.tolist(), "scale_": scaler.scale_.tolist()}
    if method == "minmax":
        if MinMaxScaler is None:
            raise ImportError("scikit-learn is required for minmax scaling.")
        scaler = MinMaxScaler().fit(X)
        return scaler.transform(X), {"method": "minmax", "data_min_": scaler.data_min_.tolist(), "data_max_": scaler.data_max_.tolist()}
    raise ValueError(f"Unknown scaling method: {method!r}")


def load_smiles_from_dataset(dataset: str, root: str, limit_n: Optional[int] = None) -> List[str]:
    ds_name = dataset.strip().upper()
    smiles: List[str] = []
    if ds_name == "ESOL":
        if MoleculeNet is None:
            raise ImportError("torch_geometric is required to load ESOL.")
        ds = MoleculeNet(root=root, name="ESOL")
        for d in ds:
            smiles.append(getattr(d, "smiles", "") or "")
        return smiles
    if ds_name == "QM9":
        if PYG_QM9 is None:
            raise ImportError("torch_geometric is required to load QM9.")
        ds = PYG_QM9(root=root)
        if limit_n is not None:
            ds = ds[: int(limit_n)]
        for d in ds:
            smiles.append(getattr(d, "smiles", "") or "")
        return smiles
    raise ValueError(f"Unsupported dataset: {dataset!r}. Choose 'ESOL' or 'QM9'.")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_npz(out_path: str, X: np.ndarray, colnames: List[str], meta: Dict[str, Any]) -> str:
    ensure_dir(os.path.dirname(out_path))
    np.savez_compressed(out_path, X=X)
    sidecar = {"columns": colnames, "meta": meta, "shape": {"N": int(X.shape[0]), "D": int(X.shape[1])}}
    with open(out_path + ".json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    return out_path


def build_descriptor_matrix(smiles: Sequence[str], cfg: DescriptorMatrixConfig) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    X_d, names_d, valid_mask = compute_rdkit_descriptors(smiles)
    if cfg.use_morgan:
        X_f, names_f = compute_morgan_fingerprints(
            smiles, radius=cfg.morgan_radius, n_bits=cfg.morgan_bits, use_chirality=cfg.morgan_use_chirality
        )
        X = np.concatenate([X_d, X_f], axis=1)
        colnames = names_d + names_f
    else:
        X = X_d
        colnames = names_d

    X = sanitize_matrix(X, replace_nan_inf_with=0.0)

    keep_mask = None
    if cfg.drop_constant:
        X, colnames, keep_mask = drop_constant_columns(X, colnames)

    X_scaled, scale_meta = scale_matrix(X, method=cfg.scale)

    meta = {
        "dataset": cfg.dataset,
        "root": cfg.root,
        "limit_n": cfg.limit_n,
        "use_morgan": cfg.use_morgan,
        "morgan_radius": cfg.morgan_radius,
        "morgan_bits": cfg.morgan_bits,
        "morgan_use_chirality": cfg.morgan_use_chirality,
        "drop_constant": cfg.drop_constant,
        "scale": cfg.scale,
        "seed": cfg.seed,
        "valid_smiles": int(np.sum(np.array(valid_mask, dtype=int))),
        "total_smiles": len(smiles),
        "kept_columns": int(X_scaled.shape[1]),
        "dropped_columns": (0 if keep_mask is None else int((~keep_mask).sum())),
        "scale_meta": scale_meta,
    }
    return X_scaled.astype(np.float32, copy=False), colnames, meta


def run_and_save(cfg: DescriptorMatrixConfig) -> str:
    smiles = load_smiles_from_dataset(cfg.dataset, cfg.root, limit_n=cfg.limit_n)
    X, colnames, meta = build_descriptor_matrix(smiles, cfg)
    ts = time.strftime("%Y%m%d-%H%M%S")
    prefix = (cfg.file_prefix or cfg.dataset.upper())
    fname = f"{prefix}_N{len(smiles)}_D{len(colnames)}_{ts}.npz"
    out_path = os.path.join(cfg.out_dir, fname)
    save_npz(out_path, X, colnames, meta)
    if cfg.verbose:
        print(f"[descriptors] Saved: {out_path} (shape={X.shape[0]}x{X.shape[1]})")
        print(f"[descriptors] Sidecar: {out_path}.json")
    return out_path


# -----------------------------------------------
# CLI
# -----------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute RDKit descriptors (+optional Morgan) and save to NPZ.")
    p.add_argument("--dataset", type=str, default="ESOL", choices=["ESOL", "QM9"])
    p.add_argument("--root", type=str, default="data")
    p.add_argument("--limit-n", type=int, default=15000, help="QM9 subsample size; -1 for full")
    p.add_argument("--no-fp", action="store_true", help="Disable Morgan fingerprints")
    p.add_argument("--fp-bits", type=int, default=1024)
    p.add_argument("--fp-radius", type=int, default=2)
    p.add_argument("--fp-no-chiral", action="store_true", help="Disable chirality in FP")
    p.add_argument("--drop-constant", action="store_true")
    p.add_argument("--scale", type=str, default="none", choices=["none", "standard", "minmax"])
    p.add_argument("--out-dir", type=str, default="outputs/descriptors")
    p.add_argument("--file-prefix", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    return p


def _cfg_from_args(args: argparse.Namespace) -> DescriptorMatrixConfig:
    limit_n = None if (args.limit_n is None or int(args.limit_n) < 0) else int(args.limit_n)
    return DescriptorMatrixConfig(
        dataset=args.dataset,
        root=args.root,
        limit_n=limit_n,
        use_morgan=(not args.no_fp),
        morgan_radius=int(args.fp_radius),
        morgan_bits=int(args.fp_bits),
        morgan_use_chirality=(not args.fp_no_chiral),
        drop_constant=bool(args.drop_constant),
        scale=args.scale,
        out_dir=args.out_dir,
        file_prefix=args.file_prefix,
        seed=int(args.seed),
        verbose=(not args.quiet),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    cfg = _cfg_from_args(args)
    try:
        run_and_save(cfg)
    except Exception as e:  # pragma: no cover
        print(f"[descriptors] ERROR: {e}", flush=True)
        return 1
    return 0


# -----------------------------------------------
# Smoke
# -----------------------------------------------

def _smoke() -> None:  # pragma: no cover
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN(CC)CC"]
    f1 = build_featurizer(kind="morgan", morgan_bits=512, morgan_radius=2)
    X1 = f1(smiles)
    assert X1.dtype == np.float32 and X1.shape == (4, 512)

    f2 = build_featurizer(kind="physchem", physchem_set="basic")
    X2 = f2(smiles)
    assert X2.dtype == np.float32 and X2.shape[0] == 4 and X2.shape[1] >= 10

    f3 = build_featurizer(kind="morgan_physchem", morgan_bits=128)
    X3 = f3(smiles)
    assert X3.dtype == np.float32 and X3.shape == (4, 128 + X2.shape[1])

    print("[descriptors] smoke OK:", X1.shape, X2.shape, X3.shape)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
