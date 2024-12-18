import torch
from AixrOptima.qoptimizer import QuantumInspiredOptimizer

def test_qoptimizer():
    param = torch.nn.Parameter(torch.randn(10, requires_grad=True))
    optimizer = QuantumInspiredOptimizer([param], lr=1e-3)
    loss = (param**2).mean()
    loss.backward()
    old_val = param.detach().clone()
    optimizer.step(loss)
    # param'ın değişmiş olması beklenir
    assert not torch.allclose(old_val, param.detach())