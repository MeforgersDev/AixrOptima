import torch

def regularization_loss(modules, lambda_reg=1e-4):
    reg_loss = 0.0
    for m in modules:
        if hasattr(m, 'lora_module'):
            reg_loss += m.lora_module.regularization_loss(lambda_reg=lambda_reg)
    return reg_loss