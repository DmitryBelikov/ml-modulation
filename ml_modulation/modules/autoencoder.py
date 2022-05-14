import torch.nn as nn
import torch.nn.functional as F

from .util import EntropyNormalization
from .modulator import Modulator


class Encoder(nn.Module):
    def __init__(self, class_count, encoding_shape):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(class_count, 4 * class_count),
            nn.ReLU(),
            nn.Linear(4 * class_count, encoding_shape),
            EntropyNormalization()
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, class_count, encoding_shape):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(encoding_shape, 4 * class_count),
            nn.ReLU(),
            nn.Linear(4 * class_count, class_count),
        )

    def forward(self, x):
        return self.decoder(x)


def NNModulator(bit_count, noise):
    encoder = Encoder(2 ** bit_count, 2)
    decoder = Decoder(2 ** bit_count, 2)
    return Modulator(encoder, decoder, noise)
