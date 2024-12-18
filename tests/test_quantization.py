import torch
from AixrOptima.quantization import QuantizationModule

def test_quantization():
    W = torch.randn(10, 20)
    q = QuantizationModule(bits=4)
    W_hat = q(W)
    assert W_hat.shape == W.shape