# tests/test_models_mpnn.py
import pytest
import torch
from torch_geometric.data import Data, Batch

from src.models.mpnn import MPNNConfig, MPNNNet


def _tiny_graph_pair(in_dim=10, edge_dim=6):
    torch.manual_seed(0)
    # Graph 1 (triangle, undirected via both directions)
    x1 = torch.randn(3, in_dim)
    ei1 = torch.tensor([[0, 1, 2, 1, 2, 0],
                        [1, 2, 0, 0, 1, 2]], dtype=torch.long)
    e1 = torch.randn(ei1.size(1), edge_dim)

    # Graph 2 (line)
    x2 = torch.randn(4, in_dim)
    ei2 = torch.tensor([[0, 1, 2],
                        [1, 2, 3]], dtype=torch.long)
    e2 = torch.randn(ei2.size(1), edge_dim)

    d1 = Data(x=x1, edge_index=ei1, edge_attr=e1)
    d2 = Data(x=x2, edge_index=ei2, edge_attr=e2)
    return Batch.from_data_list([d1, d2])


@pytest.mark.parametrize("pool", ["sum", "mean", "max"])
@pytest.mark.parametrize("use_gru", [True, False])
def test_mpnn_forward_shapes_and_finiteness(pool, use_gru):
    batch = _tiny_graph_pair(in_dim=12, edge_dim=5)
    cfg = MPNNConfig(
        in_dim=12, edge_dim=5, out_dim=3,
        hidden_dim=64, num_layers=3, pool=pool, use_gru=use_gru,
        dropout=0.0, batch_norm=True, residual=True
    )
    model = MPNNNet(cfg).eval()
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (2, 3)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("aggr", ["add", "mean", "max"])
def test_mpnn_aggr_variants(aggr):
    batch = _tiny_graph_pair(in_dim=10, edge_dim=4)
    cfg = MPNNConfig(
        in_dim=10, edge_dim=4, out_dim=1,
        hidden_dim=48, num_layers=2, aggr=aggr, pool="mean",
        dropout=0.0, use_gru=True
    )
    model = MPNNNet(cfg).eval()
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()


def test_mpnn_requires_edge_attr():
    # Missing edge_attr should raise
    x = torch.randn(3, 8)
    ei = torch.tensor([[0, 1, 2, 1, 2, 0],
                       [1, 2, 0, 0, 1, 2]], dtype=torch.long)
    d = Data(x=x, edge_index=ei)  # no edge_attr
    batch = Batch.from_data_list([d])
    cfg = MPNNConfig(in_dim=8, edge_dim=6, out_dim=1)
    model = MPNNNet(cfg)
    with pytest.raises(ValueError):
        model(batch)


def test_mpnn_backward_and_grads_flow():
    torch.manual_seed(123)
    batch = _tiny_graph_pair(in_dim=14, edge_dim=7)
    cfg = MPNNConfig(
        in_dim=14, edge_dim=7, out_dim=1,
        hidden_dim=32, num_layers=3, pool="sum", use_gru=True,
        dropout=0.0, batch_norm=True, residual=True
    )
    model = MPNNNet(cfg).train()
    out = model(batch)         # [2, 1]
    loss = (out ** 2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_mpnn_determinism_with_fixed_seed():
    torch.manual_seed(777)
    batch = _tiny_graph_pair(in_dim=10, edge_dim=6)
    cfg = MPNNConfig(
        in_dim=10, edge_dim=6, out_dim=2,
        hidden_dim=64, num_layers=2, pool="mean", use_gru=True,
        dropout=0.0
    )
    # Re-seed before each model to ensure identical params
    torch.manual_seed(9999)
    model1 = MPNNNet(cfg).eval()
    torch.manual_seed(9999)
    model2 = MPNNNet(cfg).eval()
    with torch.no_grad():
        out1 = model1(batch)
        out2 = model2(batch)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_mpnn_tiny_training_loop():
    torch.manual_seed(0)
    batch = _tiny_graph_pair(in_dim=10, edge_dim=6)
    cfg = MPNNConfig(
        in_dim=10, edge_dim=6, out_dim=1,
        hidden_dim=32, num_layers=3, pool="mean",
        use_gru=True, dropout=0.1
    )
    model = MPNNNet(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        out = model(batch)
        loss = (out ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out2 = model(batch)
    assert out2.shape == (2, 1)
    assert torch.isfinite(out2).all()
