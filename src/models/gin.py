# src/models/gin.py
"""
Graph Isomorphism Network (GIN/GINE) for Molecular Property Prediction
======================================================================

This module provides a **complete** PyTorch Geometric implementation of GIN/GINE
with optional **virtual node**, **BatchNorm**, **dropout**, **residuals**, and
a configurable **MLP readout**. It is designed to work out-of-the-box with our
ESOL and QM9 datamodules.

Key Features
------------
- GIN (node-only) or GINE (edge-aware) message passing.
- Optional **virtual node** (GIN-VN style) updated at each layer via a small MLP.
- Configurable depth, hidden size, activations, and epsilon-learnable sum-aggregator.
- Global pooling: sum/mean/max (selectable).
- MLP readout with optional dropout and normalization.
- Friendly `forward(batch)` signature expecting standard PyG Batch with
  `x`, `edge_index`, optional `edge_attr`, and `batch` graph indices.

Usage
-----
```python
from src.models.gin import GINConfig, GINNet

cfg = GINConfig(in_dim=node_dim, edge_dim=edge_dim, out_dim=1)
model = GINNet(cfg)
pred = model(batch)  # returns shape [num_graphs, out_dim]
```

References
----------
- Xu et al., "How Powerful are Graph Neural Networks?" (ICLR 2019)
- Hu et al., "Strategies for Pre-training Graph Neural Networks" (KDD 2020) – virtual node idea
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, GINEConv, global_add_pool, global_mean_pool, global_max_pool


PoolType = Literal["sum", "mean", "max"]
ActType = Literal["relu", "gelu", "leaky_relu", "elu"]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _make_activation(name: ActType) -> nn.Module:
    name = name.lower()
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
    """Build a simple MLP with optional BatchNorm and Dropout between layers."""
    layers = []
    act_fn = _make_activation(act)
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:  # hidden layer (not the last)
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(act_fn)
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


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class GINConfig:
    # Input / Output sizes
    in_dim: int
    out_dim: int = 1
    edge_dim: Optional[int] = None  # if provided and use_edge_attr=True, GINE is used

    # Architecture
    hidden_dim: int = 128
    num_layers: int = 5
    dropout: float = 0.1
    batch_norm: bool = True
    residual: bool = True
    act: ActType = "relu"
    pool: PoolType = "mean"

    # GIN/GINE options
    use_edge_attr: bool = True   # True → GINE if edge_dim is provided, else falls back to GIN
    learn_eps: bool = True       # learnable epsilon in GIN aggregation

    # Virtual node (graph-level global token) options
    virtual_node: bool = False
    vn_hidden_mult: int = 1      # vn hidden dimension as multiple of hidden_dim

    # Readout (after pooling)
    readout_layers: int = 2      # number of linear layers in readout MLP (>=1)
    readout_hidden_mult: float = 1.0  # width of readout hidden layers (multiplier of hidden_dim)

    # Init
    init: Literal["kaiming", "xavier", "none"] = "kaiming"


# -----------------------------------------------------------------------------
# Layers
# -----------------------------------------------------------------------------

class EdgeEncoder(nn.Module):
    """Linear projection for continuous edge_attr to desired hidden dim."""
    def __init__(self, in_dim: int, out_dim: int, act: ActType = "relu"):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = _make_activation(act)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_attr is None:
            raise ValueError("EdgeEncoder received None edge_attr.")
        return self.act(self.proj(edge_attr))


class VirtualNode(nn.Module):
    """
    Simple virtual node as in GIN-VN: per-graph embedding updated each layer.
    Update rule:
        v_{l+1} = v_l + MLP( readout(h_l) )
    where readout is sum pooling over nodes; v_0 initialized to zeros.
    """
    def __init__(self, hidden_dim: int, act: ActType = "relu"):
        super().__init__()
        self.updater = mlp([hidden_dim, hidden_dim], act=act, dropout=0.0, batch_norm=False)

    def forward(self, h: torch.Tensor, batch: torch.Tensor, v: torch.Tensor, pool_fn) -> torch.Tensor:
        pooled = pool_fn(h, batch)  # [num_graphs, hidden]
        return v + self.updater(pooled)


class GINBlock(nn.Module):
    """One GIN/GINE layer with (optional) BatchNorm, activation, dropout, and residual."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act: ActType = "relu",
        batch_norm: bool = True,
        dropout: float = 0.0,
        residual: bool = True,
        use_edge_attr: bool = False,
        edge_encoder: Optional[nn.Module] = None,
        learn_eps: bool = True,
    ):
        super().__init__()
        self.residual = residual and (in_dim == out_dim)
        self.batch_norm = batch_norm
        self.dropout_p = float(dropout)
        self.act = _make_activation(act)

        # Per-message MLP inside GIN(E) conv
        nn_message = mlp([out_dim, out_dim], act=act, dropout=dropout, batch_norm=True)

        if use_edge_attr and edge_encoder is not None:
            self.conv = GINEConv(nn_message, eps=0.0, train_eps=learn_eps)
            self.edge_encoder = edge_encoder
        else:
            self.conv = GINConv(nn_message, eps=0.0, train_eps=learn_eps)
            self.edge_encoder = None

        # Input projection (if required) before conv
        self.pre = None
        if in_dim != out_dim:
            self.pre = nn.Linear(in_dim, out_dim)

        self.norm = nn.BatchNorm1d(out_dim) if batch_norm else None
        self.drop = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

    def forward(self, x, edge_index, edge_attr=None):
        h = x
        if self.pre is not None:
            h = self.pre(h)
        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            h = self.conv(h, edge_index, edge_attr)
        else:
            h = self.conv(h, edge_index)
        if self.norm is not None:
            h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        if self.residual:
            h = h + (x if self.pre is None else self.pre(x))
        return h


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class GINNet(nn.Module):
    def __init__(self, cfg: GINConfig):
        super().__init__()
        self.cfg = cfg

        pool_fn = get_pool_fn(cfg.pool)
        self.pool_fn = pool_fn

        # Edge encoder if GINE is desired
        self.edge_encoder = None
        use_gine = cfg.use_edge_attr and (cfg.edge_dim is not None and cfg.edge_dim > 0)
        if use_gine:
            self.edge_encoder = EdgeEncoder(cfg.edge_dim, cfg.hidden_dim, act=cfg.act)

        # Input projection so that VN addition has matching dims
        self.input_proj = nn.Linear(cfg.in_dim, cfg.hidden_dim) if cfg.in_dim != cfg.hidden_dim else nn.Identity()

        # Build blocks: all operate at hidden_dim to keep shapes consistent
        blocks = []
        in_dim = cfg.hidden_dim
        for layer in range(cfg.num_layers):
            blocks.append(
                GINBlock(
                    in_dim=in_dim if layer == 0 else cfg.hidden_dim,
                    out_dim=cfg.hidden_dim,
                    act=cfg.act,
                    batch_norm=cfg.batch_norm,
                    dropout=cfg.dropout,
                    residual=cfg.residual,
                    use_edge_attr=use_gine,
                    edge_encoder=self.edge_encoder if use_gine else None,
                    learn_eps=cfg.learn_eps,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # Virtual node
        self.use_vn = bool(cfg.virtual_node)
        if self.use_vn:
            self.vn = VirtualNode(cfg.hidden_dim, act=cfg.act)

        # Readout MLP
        readout_dims = [cfg.hidden_dim]
        hidden_ro = int(round(cfg.hidden_dim * float(cfg.readout_hidden_mult)))
        if cfg.readout_layers <= 1:
            readout_dims.append(cfg.out_dim)
        else:
            for _ in range(cfg.readout_layers - 1):
                readout_dims.append(hidden_ro)
            readout_dims.append(cfg.out_dim)
        self.readout = mlp(readout_dims, act=cfg.act, dropout=cfg.dropout, batch_norm=cfg.batch_norm)
        # --- post-hoc calibration wiring (no-op unless attached) ---
        self._posthoc_calibrator = None          # nn.Module or None
        self._apply_calibration_on_eval = True   # apply only when not self.training

        # Init
        self.apply(self._init_weights if cfg.init != "none" else lambda m: None)

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
            batch: torch_geometric.data.Batch with attributes:
                - x: [num_nodes, in_dim] node features
                - edge_index: [2, num_edges]
                - edge_attr: [num_edges, edge_dim] (optional for GINE)
                - batch: [num_nodes] graph indices
        Returns:
            y_hat: [num_graphs, out_dim] predictions
        """
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, getattr(batch, "edge_attr", None), batch.batch

        # Ensure proper dtypes for linear layers / convs
        x = x.to(torch.float32)
        if edge_attr is not None:
            edge_attr = edge_attr.to(torch.float32)

        h = self.input_proj(x)
        vn = None
        if self.use_vn:
            # Initialize per-graph virtual node to zeros
            num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0
            vn = torch.zeros(num_graphs, self.cfg.hidden_dim, device=h.device)

        for i, block in enumerate(self.blocks):
            if self.use_vn:
                # Add virtual node embedding to each node embedding before the layer
                h = h + vn[batch_idx]
            h = block(h, edge_index, edge_attr=edge_attr)
            if self.use_vn:
                vn = self.vn(h, batch_idx, vn, self.pool_fn)

        # Pool to graph embedding
        g = self.pool_fn(h, batch_idx)  # [num_graphs, hidden_dim]

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

    # Two tiny graphs (triangle and a line)
    x1 = torch.randn(3, 10)
    ei1 = torch.tensor([[0, 1, 2, 1, 2, 0],
                        [1, 2, 0, 0, 1, 2]], dtype=torch.long)
    x2 = torch.randn(4, 10)
    ei2 = torch.tensor([[0, 1, 2],
                        [1, 2, 3]], dtype=torch.long)

    d1 = Data(x=x1, edge_index=ei1)
    d2 = Data(x=x2, edge_index=ei2)
    batch = Batch.from_data_list([d1, d2])

    cfg = GINConfig(in_dim=10, out_dim=1, edge_dim=None, use_edge_attr=False, num_layers=3, hidden_dim=32, virtual_node=True)
    model = GINNet(cfg)
    out = model(batch)
    assert out.shape == (2, 1)
    print("[gin] Smoke test passed. Output:", out.squeeze().tolist())


if __name__ == "__main__":
    _smoke_test()
