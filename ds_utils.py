import numpy as np
import torch


@torch.no_grad()
def mse(tensor1, tensor2):
    return torch.mean((tensor1 - tensor2) ** 2).item()


def mse_numpy(weights1, weights2):
    return np.mean((weights1 - weights2) ** 2)
