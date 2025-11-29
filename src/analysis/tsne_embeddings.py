# src/analysis/tsne_embeddings.py
from __future__ import annotations

"""
t-SNE embedding visualization for ESOL (Phase 7)
===============================================

This script:
  - Loads a trained GIN / MPNN checkpoint from a Phase-6 run directory.
  - Loads ESOL via torch_geometric.datasets.MoleculeNet.
  - Collects graph-level embeddings:
        g = pooled graph representation right before the final readout MLP.
    (We capture this via a forward pre-hook on model.readout.)
  - Runs t-SNE on these embeddings.
  - Colors points by target bins (quantile bins of ESOL solubility).
  - Saves a single-panel figure under report/figures/.

Usage
-----
  python -m src.analysis.tsne_embeddings \
    --run-dir runs/esol_gin_scaffold_seed0_20251128-164102 \
    --model gin \
    --dataset ESOL \
    --max-samples 1000
"""

import argparse
import os
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.models.gin import GINConfig, GINNet
from src.models.mpnn import MPNNConfig, MPNNNet


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _infer_device(device_str: str | None = None) -> torch.device:
    if device_str is not None and device_str.lower() != "auto":
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
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


def _collect_embeddings_and_targets(
    model: nn.Module,
    dataset: MoleculeNet,
    device: torch.device,
    batch_size: int = 128,
    max_samples: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        embeddings: [N, D]
        targets:    [N]
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    model.to(device)

    # We try to hook into model.readout to capture 'g' before final MLP.
    embeddings_chunks: List[torch.Tensor] = []
    targets_chunks: List[torch.Tensor] = []

    use_preds_directly = False

    handle = None
    if hasattr(model, "readout") and isinstance(model.readout, nn.Module):
        def _pre_hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
            # inputs[0] is g: [B, H]
            g = inputs[0]
            embeddings_chunks.append(g.detach().cpu())

        handle = model.readout.register_forward_pre_hook(_pre_hook)
    else:
        # Fallback: treat model output as embedding.
        use_preds_directly = True

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            if use_preds_directly:
                out = model(batch)
                if isinstance(out, torch.Tensor):
                    if out.dim() == 1:
                        g = out.unsqueeze(-1)
                    else:
                        g = out
                else:
                    raise RuntimeError(f"Unexpected model output type: {type(out)}")
                embeddings_chunks.append(g.detach().cpu())
            else:
                # Trigger the hook; we ignore the actual predictions.
                _ = model(batch)

            y = batch.y.view(-1).detach().cpu()
            targets_chunks.append(y)

    if handle is not None:
        handle.remove()

    if not embeddings_chunks:
        raise RuntimeError("No embeddings were collected; check model.readout / hook logic.")

    embeddings = torch.cat(embeddings_chunks, dim=0)
    targets = torch.cat(targets_chunks, dim=0)

    if max_samples is not None and embeddings.size(0) > max_samples:
        rng = np.random.RandomState(0)
        idx = rng.choice(embeddings.size(0), size=max_samples, replace=False)
        embeddings = embeddings[idx]
        targets = targets[idx]

    return embeddings.numpy(), targets.numpy()


def _bin_targets_quantile(y: np.ndarray, n_bins: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin targets into roughly equal-sized quantile bins.

    Returns:
        bin_ids: [N] integer bin index
        bin_edges: [n_bins+1] array of edges
    """
    y = y.astype(float).reshape(-1)
    # small safety margin if too few points
    n_bins = max(2, min(n_bins, max(2, y.shape[0] // 10)))
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(y, quantiles)
    # Make edges strictly increasing if necessary
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    bin_ids = np.digitize(y, edges[1:-1], right=True)
    return bin_ids, edges


def _run_tsne(
    X: np.ndarray,
    perplexity: float = 30.0,
    random_state: int = 0,
) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
        random_state=random_state,
    )
    return tsne.fit_transform(X)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_tsne_embeddings(
    run_dir: str,
    model_name: str,
    dataset_name: str = "ESOL",
    data_root: str = "data",
    ckpt_name: str = "best.ckpt",
    max_samples: int | None = 2000,
    perplexity: float = 30.0,
    seed: int = 0,
    out_fig: str | None = None,
    device_str: str | None = None,
) -> str:
    device = _infer_device(device_str)
    run_dir = os.path.abspath(run_dir)
    ckpt_path = os.path.join(run_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    dataset = _build_dataset(dataset_name, data_root)
    model = _build_model(model_name, dataset)
    _load_checkpoint(model, ckpt_path, device)

    # Collect embeddings + targets
    embeddings, targets = _collect_embeddings_and_targets(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=128,
        max_samples=max_samples,
    )

    # Target bins
    bin_ids, edges = _bin_targets_quantile(targets, n_bins=5)

    # t-SNE
    emb_2d = _run_tsne(embeddings, perplexity=perplexity, random_state=seed)

    # Prepare figure
    figs_dir = os.path.join("report", "figures")
    _ensure_dir(figs_dir)
    if out_fig is None:
        out_fig = os.path.join(
            figs_dir, f"tsne_embeddings_{dataset_name}_{model_name}.png"
        )

    plt.figure(figsize=(6, 5))
    cmap = plt.cm.get_cmap("viridis", int(bin_ids.max()) + 1)
    scatter = plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=bin_ids,
        cmap=cmap,
        s=12,
        alpha=0.8,
    )
    cbar = plt.colorbar(scatter)
    # Nice labels for bins
    tick_positions = np.arange(int(bin_ids.max()) + 1)
    tick_labels: List[str] = []
    for k in tick_positions:
        lo = edges[k]
        hi = edges[k + 1]
        tick_labels.append(f"[{lo:.2f}, {hi:.2f}]")
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    cbar.set_label("ESOL target bin")

    plt.title(f"t-SNE of graph embeddings — {dataset_name} ({model_name.upper()})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.close()

    return out_fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="t-SNE visualization of graph-level embeddings for ESOL."
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
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum number of graphs to subsample for t-SNE (default: 2000).",
    )
    p.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (default: 30.0).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for t-SNE.",
    )
    p.add_argument(
        "--out-fig",
        type=str,
        default=None,
        help="Optional output figure path (default: report/figures/tsne_embeddings_*.png).",
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
    fig_path = run_tsne_embeddings(
        run_dir=args.run_dir,
        model_name=args.model,
        dataset_name=args.dataset,
        data_root=args.data_root,
        ckpt_name=args.ckpt_name,
        max_samples=args.max_samples,
        perplexity=args.perplexity,
        seed=args.seed,
        out_fig=args.out_fig,
        device_str=args.device,
    )
    print(f"[tsne_embeddings] Wrote figure -> {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
