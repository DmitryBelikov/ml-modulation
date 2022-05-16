import numpy as np
import torch
import pytorch_lightning as pl


class AwgnNoise(pl.LightningModule):
    def __init__(self, snr):
        super().__init__()
        self.sigma = np.sqrt(1 / (2 * 10 ** (snr / 10)))

    def forward(self, x, multiplier=1.):
        return x + multiplier * torch.normal(torch.zeros_like(x, device=self.device),
                                             torch.full_like(x, self.sigma, device=self.device))


class Clipper(pl.LightningModule):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def forward(self, data):
        min_bound = torch.tensor([-self.x, -self.y], device=self.device)
        max_bound = torch.tensor([self.x, self.y], device=self.device)
        return torch.clip(data, min_bound, max_bound)


class ClippingNoise(pl.LightningModule):
    def __init__(self, snr, x=0.5, y=1000.):
        super().__init__()
        self.awgn = AwgnNoise(snr)
        self.clipper = Clipper(x, y)

    def forward(self, x):
        x = self.awgn(x)
        return self.clipper(x)
