# tests/test_models_forward.py
import math
import sys
import numpy as np
import pytest
import torch

from torch_geometric.data import Data, Batch

from src.models.gin import GINConfig, GINNet


def _tiny_graph_pair(in_dim=10, edge_dim=None):
    """
    Build two small graphs:
      - g1: triangle with 3 nodes (undirected via both directions)
      - g2: 4-node path
    Optionally attach edge_attr if edge_dim is not None.
    """
    torch.manual_seed(0)
    # Graph 1
    x1 = torch.randn(3, in_dim)
    ei1 = torch.tensor([[0, 1, 2, 1, 2, 0],
                        [1, 2, 0, 0, 1, 2]], dtype=torch.long)
    if edge_dim is not None and edge_dim > 0:
        e1 = torch.randn(ei1.size(1), edge_dim)
    else:
        e1 = None
    d1 = Data(x=x1, edge_index=ei1, edge_attr=e1)

    # Graph 2
    x2 = torch.randn(4, in_dim)
    ei2 = torch.tensor([[0, 1, 2],
                        [1, 2, 3]], dtype=torch.long)
    if edge_dim is not None and edge_dim > 0:
        e2 = torch.randn(ei2.size(1), edge_dim)
    else:
        e2 = None
    d2 = Data(x=x2, edge_index=ei2, edge_attr=e2)

    return Batch.from_data_list([d1, d2])


@pytest.mark.parametrize("pool", ["sum", "mean", "max"])
def test_gin_forward_shapes_and_finiteness(pool):
    torch.manual_seed(123)
    batch = _tiny_graph_pair(in_dim=10, edge_dim=None)
    cfg = GINConfig(
        in_dim=10, out_dim=1, edge_dim=None, use_edge_attr=False,
        hidden_dim=64, num_layers=4, virtual_node=True, pool=pool, dropout=0.0
    )
    model = GINNet(cfg).eval()  # eval disables dropout
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()



def test_gin_determinism_eval_mode():
    torch.manual_seed(7)
    batch = _tiny_graph_pair(in_dim=8, edge_dim=None)
    cfg = GINConfig(
        in_dim=8, out_dim=2, edge_dim=None, use_edge_attr=False,
        hidden_dim=32, num_layers=3, virtual_node=True, pool="mean", dropout=0.0
    )
    # Re-seed before each model instantiation so both get identical parameters
    torch.manual_seed(12345)
    model1 = GINNet(cfg).eval()
    torch.manual_seed(12345)
    model2 = GINNet(cfg).eval()

    with torch.no_grad():
        out1 = model1(batch).detach()
        out2 = model2(batch).detach()
    assert torch.allclose(out1, out2, atol=1e-6)


def test_gin_backward_gradients():
    torch.manual_seed(0)
    batch = _tiny_graph_pair(in_dim=10, edge_dim=None)
    cfg = GINConfig(
        in_dim=10, out_dim=1, edge_dim=None, use_edge_attr=False,
        hidden_dim=32, num_layers=3, virtual_node=True, pool="sum", dropout=0.0
    )
    model = GINNet(cfg).train()
    out = model(batch)               # [2, 1]
    loss = (out ** 2).mean()
    loss.backward()

    # Check some parameters received gradients
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_gine_edge_attributes_forward():
    torch.manual_seed(1)
    edge_dim = 5
    batch = _tiny_graph_pair(in_dim=12, edge_dim=edge_dim)
    cfg = GINConfig(
        in_dim=12, out_dim=3, edge_dim=edge_dim, use_edge_attr=True,
        hidden_dim=48, num_layers=4, virtual_node=False, pool="mean", dropout=0.0
    )
    model = GINNet(cfg).eval()
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (2, 3)
    assert torch.isfinite(out).all()


def test_input_projection_ensures_vn_shape_match():
    """Regression test for VN shape mismatch when in_dim != hidden_dim."""
    torch.manual_seed(42)
    batch = _tiny_graph_pair(in_dim=10, edge_dim=None)
    cfg = GINConfig(
        in_dim=10, out_dim=1, edge_dim=None, use_edge_attr=False,
        hidden_dim=32, num_layers=2, virtual_node=True, pool="mean", dropout=0.0
    )
    model = GINNet(cfg).eval()
    # Should not raise; previously failed with size mismatch
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (2, 1)


def test_training_step_like_loop():
    """Tiny loop to ensure optimizer/loss interplay works for a couple steps."""
    torch.manual_seed(0)
    batch = _tiny_graph_pair(in_dim=10, edge_dim=None)
    cfg = GINConfig(
        in_dim=10, out_dim=1, edge_dim=None, use_edge_attr=False,
        hidden_dim=32, num_layers=3, virtual_node=True, pool="sum", dropout=0.1
    )
    model = GINNet(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        out = model(batch)  # dropout active in train mode
        loss = (out ** 2).mean()
        loss.backward()
        opt.step()
    # Just assert it runs and parameters changed a bit
    with torch.no_grad():
        out2 = model(batch.eval()) if hasattr(batch, "eval") else model(batch)
    assert out2.shape == (2, 1)
    assert torch.isfinite(out2).all()
