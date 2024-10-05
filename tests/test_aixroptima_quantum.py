import numpy as np
import sys
import os
from aixroptima_quantum import AixrOptimaQuantum

def loss_fn(params):
    """Sample Loss."""
    return np.sum(params ** 2)

def test_aixroptima_update():
    params = np.array([0.5, -0.5, 0.1])
    optimizer = AixrOptimaQuantum(params)
    
    grads = np.array([0.1, -0.2, 0.05])
    updated_params = optimizer.update(grads)
    
    assert updated_params is not None
    print("Test passed: Parameters successfully updated.")

if __name__ == "__main__":
    test_aixroptima_update()
