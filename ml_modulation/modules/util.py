import torch.nn as nn


class EntropyNormalization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        energy = (x ** 2).sum(axis=1).mean()
        return x / energy.sqrt()
