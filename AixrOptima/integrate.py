import torch
from .lora import LowRankAdaptation

def integrate_aixroptima(model, rank=4, bits=4, use_quantization=True, per_channel_quant=True):
    """
    Modelin linear katmanlarını tespit ederek LowRankAdaptation ekler.
    Orijinal ağırlıkları dondurur, forward'da LowRankAdaptation'dan elde edilen
    ağırlıkları kullanır.
    """
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None and module.weight.ndim == 2:
            with torch.no_grad():
                original_weight = module.weight.data
            lora_module = LowRankAdaptation(original_weight, rank=rank, quant_bits=bits,
                                            use_quantization=use_quantization,
                                            per_channel_quant=per_channel_quant)
            module.weight.requires_grad = False
            def replace_weight_forward(m, inp, out):
                x = inp[0]
                new_W = lora_module()
                bias = m.bias
                return x.matmul(new_W.T) + (bias if bias is not None else 0)
            module.register_forward_hook(replace_weight_forward)
            module.lora_module = lora_module
    return model