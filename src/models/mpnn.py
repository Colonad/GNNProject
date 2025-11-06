# src/models/mpnn.py
"""
Message Passing Neural Network (MPNN) with NNConv for Molecular Graphs
======================================================================

This module implements a strong MPNN-style architecture using
**torch_geometric.nn.NNConv** with continuous **edge attributes** (bond features).
It is designed for ESOL/QM9 datamodules and mirrors the ergonomics of `GINNet`.

Key features
------------
- Stacked **NNConv** layers (edge-conditioned linear transformations).
- Edge-network MLPs (one per layer) that map `edge_attr -> weight matrix`.
- Optional **GRU** state updates after each message passing step.
- Configurable depth, hidden size, activations, normalization, dropout, residuals.
- Global pooling: `sum` / `mean` / `max`.
- Detailed parameter count logging (Definition of Done).

Usage
-----
```python
from src.models.mpnn import MPNNConfig, MPNNNet

cfg = MPNNConfig(in_dim=node_dim, edge_dim=edge_dim, out_dim=1)
model = MPNNNet(cfg)
y_hat = model(batch)  # [num_graphs, out_dim]
```

References
----------
- Gilmer et al., "Neural Message Passing for Quantum Chemistry", ICML 2017.
- PyG: `torch_geometric.nn.NNConv` example (edge-conditioned conv + GRU).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import NNConv, global_add_pool, global_mean_pool, global_max_pool


PoolType = Literal["sum", "mean", "max"]
ActType = Literal["relu", "gelu", "leaky_relu", "elu"]
AggrType = Literal["add", "mean", "max"]


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def _make_activation(name: ActType) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    raise ValueError(f"Unknown activation: {name}")


def mlp(dims: Sequence[int], act: ActType = "relu", dropout: float = 0.0, batch_norm: bool = True) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(_make_activation(act))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def get_pool_fn(name: PoolType):
    name = name.lower()
    if name == "sum":
        return global_add_pool
    if name == "mean":
        return global_mean_pool
    if name == "max":
        return global_max_pool
    raise ValueError(f"Unknown pool type: {name}")


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class MPNNConfig:
    # IO sizes
    in_dim: int
    edge_dim: int
    out_dim: int = 1

    # Architecture
    hidden_dim: int = 128
    num_layers: int = 6
    aggr: AggrType = "add"            # aggregation inside NNConv
    pool: PoolType = "mean"
    act: ActType = "relu"
    dropout: float = 0.1
    batch_norm: bool = True
    residual: bool = True

    # GRU-based state update (recommended for NNConv-based MPNN)
    use_gru: bool = True

    # Readout
    readout_layers: int = 2
    readout_hidden_mult: float = 1.0

    # Init
    init: Literal["kaiming", "xavier", "none"] = "kaiming"

    # Logging
    verbose: bool = True


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class MPNNNet(nn.Module):
    def __init__(self, cfg: MPNNConfig):
        super().__init__()
        self.cfg = cfg
        self.pool_fn = get_pool_fn(cfg.pool)

        # Input projection to hidden dim
        self.input_proj = nn.Linear(cfg.in_dim, cfg.hidden_dim) if cfg.in_dim != cfg.hidden_dim else nn.Identity()

        # Edge networks (one per layer): map edge_attr -> (hidden_dim * hidden_dim) weight vector
        self.edge_mlps = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drops = nn.ModuleList()

        for layer in range(cfg.num_layers):
            edge_mlp = mlp(
                dims=[cfg.edge_dim, cfg.hidden_dim * 2, cfg.hidden_dim * cfg.hidden_dim],
                act=cfg.act,
                dropout=cfg.dropout * 0.5,
                batch_norm=False,  # batch-norm on edge features not always stable
            )
            self.edge_mlps.append(edge_mlp)

            conv = NNConv(
                in_channels=cfg.hidden_dim,
                out_channels=cfg.hidden_dim,
                nn=edge_mlp,
                aggr=cfg.aggr,
                root_weight=True,
                bias=True,
            )
            self.convs.append(conv)

            self.norms.append(nn.BatchNorm1d(cfg.hidden_dim) if cfg.batch_norm else nn.Identity())
            self.drops.append(nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity())

        # Optional GRU for iterative state updates
        self.gru = nn.GRU(cfg.hidden_dim, cfg.hidden_dim) if cfg.use_gru else None





        # Readout MLP
        ro_hidden = int(round(cfg.hidden_dim * float(cfg.readout_hidden_mult)))
        ro_dims = [cfg.hidden_dim]
        if cfg.readout_layers <= 1:
            ro_dims.append(cfg.out_dim)
        else:
            for _ in range(cfg.readout_layers - 1):
                ro_dims.append(ro_hidden)
            ro_dims.append(cfg.out_dim)
        self.readout = mlp(ro_dims, act=cfg.act, dropout=cfg.dropout, batch_norm=cfg.batch_norm)


        # --- post-hoc calibration wiring (no-op unless attached) ---
        self._posthoc_calibrator = None          # nn.Module or None
        self._apply_calibration_on_eval = True   # apply only when not self.training




        # Init weights
        if cfg.init != "none":
            self.apply(self._init_weights)

        # Parameter count logging (DoD)
        if cfg.verbose:
            total = count_parameters(self, trainable_only=True)
            print(f"[mpnn] parameters: {total:,d} (trainable)")

    # weight init
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            if self.cfg.init == "kaiming":
                nn.init.kaiming_uniform_(m.weight, a=0.1)
            elif self.cfg.init == "xavier":
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


    # --- public API for calibrators ---
    def set_calibrator(self, module, apply_on_eval: bool = True) -> None:
        """
        Attach a post-hoc calibrator (e.g., TemperatureScaler).
        Only used when the network outputs class logits (C > 1).
        """
        self._posthoc_calibrator = module
        self._apply_calibration_on_eval = bool(apply_on_eval)

    def clear_calibrator(self) -> None:
        self._posthoc_calibrator = None

    def get_calibrator(self):
        return self._posthoc_calibrator

    @torch.no_grad()
    def probs(self, *args, **kwargs):
        """
        Convenience: forward pass → probabilities if output looks like multi-class logits.
        Falls back to raw output otherwise.
        """
        out = self.forward(*args, **kwargs)
        if isinstance(out, torch.Tensor) and out.dim() >= 2 and out.size(-1) > 1:
            return torch.softmax(out, dim=-1)
        return out




    def forward(self, batch) -> torch.Tensor:
        """
        Args:
            batch: PyG Batch with fields:
                - x: [num_nodes, in_dim]
                - edge_index: [2, num_edges]
                - edge_attr: [num_edges, edge_dim]  (REQUIRED: uses bond features)
                - batch: [num_nodes] graph IDs
        Returns:
            y_hat: [num_graphs, out_dim]
        """
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, getattr(batch, "edge_attr", None), batch.batch
        if edge_attr is None:
            raise ValueError("MPNN requires edge_attr; got None. Provide bond features (type/order/aromatic).")

        h = self.input_proj(x)  # [N, hidden]
        h0 = h

        # For GRU update we need shape [1, N, hidden]
        if self.gru is not None:
            h_gru = h.unsqueeze(0)  # (1, N, H)

        for l, (conv, norm, drop) in enumerate(zip(self.convs, self.norms, self.drops)):
            m = conv(h, edge_index, edge_attr)  # message -> [N, H]
            if self.gru is not None:
                # GRU expects (seq_len, batch, input_size); here use seq_len=1 with N as batch dimension
                m_seq = m.unsqueeze(0)
                h_gru, _ = self.gru(m_seq, h_gru)
                h = h_gru.squeeze(0)
            else:
                h = m

            # Post-processing
            if isinstance(norm, nn.BatchNorm1d):
                h = norm(h)
            h = _make_activation(self.cfg.act)(h)
            h = drop(h)

            # Residual connection (skip from input of the layer)
            if self.cfg.residual:
                h = h + h0 if l == 0 else h + h_prev

            h_prev = h

        # Pool to graph-level
        g = get_pool_fn(self.cfg.pool)(h, batch_idx)  # [num_graphs, H]

        # Readout
        out = self.readout(g)  # [num_graphs, out_dim]

        # --- apply calibrator only in eval and only for multi-class logits ---
        if (
            not self.training
            and self._posthoc_calibrator is not None
            and self._apply_calibration_on_eval
            and isinstance(out, torch.Tensor)
            and out.dim() >= 2
            and out.size(-1) > 1
        ):
            out = self._posthoc_calibrator(out)

        return out



# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

def _smoke_test():
    import torch
    from torch_geometric.data import Data, Batch

    torch.manual_seed(0)

    # Build two small graphs with edge attributes (e.g., 6-d bond feats)
    edge_dim = 6
    x1 = torch.randn(3, 10)
    ei1 = torch.tensor([[0, 1, 2, 1, 2, 0],
                        [1, 2, 0, 0, 1, 2]], dtype=torch.long)
    e1 = torch.randn(ei1.size(1), edge_dim)

    x2 = torch.randn(4, 10)
    ei2 = torch.tensor([[0, 1, 2],
                        [1, 2, 3]], dtype=torch.long)
    e2 = torch.randn(ei2.size(1), edge_dim)

    d1 = Data(x=x1, edge_index=ei1, edge_attr=e1)
    d2 = Data(x=x2, edge_index=ei2, edge_attr=e2)
    batch = Batch.from_data_list([d1, d2])

    cfg = MPNNConfig(in_dim=10, edge_dim=edge_dim, out_dim=1, hidden_dim=64, num_layers=4, use_gru=True, pool="mean")
    model = MPNNNet(cfg)
    out = model(batch)
    assert out.shape == (2, 1)
    print("[mpnn] Smoke test passed. Output:", out.squeeze().tolist())

if __name__ == "__main__":
    _smoke_test()
