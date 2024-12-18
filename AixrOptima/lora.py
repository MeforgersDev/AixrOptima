import torch
import torch.nn as nn
from .quantization import QuantizationModule

class LowRankAdaptation(nn.Module):
    def __init__(self, original_weight, rank=4, quant_bits=4, use_quantization=True, per_channel_quant=False):
        super(LowRankAdaptation, self).__init__()
        self.use_quantization = use_quantization
        self.quantizer = QuantizationModule(bits=quant_bits, per_channel=per_channel_quant)

        # Orijinal ağırlıkları dondur
        self.W_base = nn.Parameter(original_weight.detach().clone(), requires_grad=False)

        out_dim, in_dim = self.W_base.shape
        self.A = nn.Parameter(torch.randn(out_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, in_dim) * 0.01)

    def forward(self):
        W_new = self.W_base + torch.matmul(self.A, self.B)
        if self.use_quantization:
            W_new = self.quantizer(W_new)
        return W_new

    def regularization_loss(self, lambda_reg=1e-4):
        # Basit L2 düzenlileştirme örneği
        return lambda_reg * (self.A.norm(p=2)**2 + self.B.norm(p=2)**2)