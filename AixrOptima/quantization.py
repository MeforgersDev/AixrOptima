import torch
import torch.nn as nn

class QuantizationModule(nn.Module):
    def __init__(self, bits=4, per_channel=False):
        super().__init__()
        self.bits = bits
        self.per_channel = per_channel
        
    def forward(self, W):
        qmax = 2 ** self.bits - 1
        if self.per_channel:
            # Per-channel quantization (assume W shape [out_dim, in_dim])
            W_min = W.min(dim=1, keepdim=True)[0]
            W_max = W.max(dim=1, keepdim=True)[0]
        else:
            W_min = W.min()
            W_max = W.max()

        scale = (W_max - W_min) / (qmax + 1e-8)
        W_q = torch.round((W - W_min) / (scale + 1e-8)).clamp(0, qmax)
        W_hat = W_q * scale + W_min
        return W_hat