import torch
from AixrOptima.lora import LowRankAdaptation

def test_lora():
    W = torch.randn(10, 20)
    lora = LowRankAdaptation(W, rank=4, quant_bits=4, use_quantization=False)
    W_new = lora()
    assert W_new.shape == W.shape
