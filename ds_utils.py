import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def mse(weights1, weights2):
    return torch.mean((weights1 - weights2) ** 2)

def mse_numpy(weights1, weights2):
    return np.mean((weights1 - weights2) ** 2)