import math
import torch

class QuantumInspiredOptimizer:
    def __init__(self, params, lr=1e-3, initial_temp=1.0, cooling_rate=0.99):
        self.params = list(params)
        self.lr = lr
        self.temp = initial_temp
        self.cooling_rate = cooling_rate

    def step(self, loss):
        grads = torch.autograd.grad(loss, self.params, create_graph=False)

        old_loss = loss.item()
        old_state = [p.clone() for p in self.params]

        new_params = []
        for p, g in zip(self.params, grads):
            new_p = p - self.lr * g
            noise = torch.randn_like(p) * self.temp * 0.01
            new_p = new_p + noise
            new_params.append(new_p)

        with torch.no_grad():
            for p, np in zip(self.params, new_params):
                p.data = np.data

        new_loss_val = loss.detach().item()
        delta = new_loss_val - old_loss
        accept_prob = math.exp(-delta/(self.temp+1e-8))
        if delta > 0.0 and torch.rand(1).item() > accept_prob:
            with torch.no_grad():
                for p, op in zip(self.params, old_state):
                    p.data = op.data

        self.temp = self.temp * self.cooling_rate