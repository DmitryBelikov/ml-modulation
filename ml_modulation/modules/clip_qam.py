import torch
import torch.nn as nn
import torch.nn.functional as F

from .qam import qam_mapping
from .modulator import Modulator


def clip_qam_mapping(bit_count, x):
    qam = qam_mapping(bit_count)
    max_x = qam[:, 0].abs().max()
    clip_qam = torch.clone(qam)
    clip_qam[:, 0] *= (x / max_x)
    x_energy = (clip_qam[:, 0] ** 2).sum()
    cur_y_energy = (clip_qam[:, 1] ** 2).sum()
    result_y_energy = 2 ** bit_count - x_energy
    clip_qam[:, 1] *= torch.sqrt(result_y_energy / cur_y_energy)
    return clip_qam


class ClipQamEncoder(nn.Module):
    def __init__(self, bit_count, x=.5):
        super().__init__()
        self.mapping = clip_qam_mapping(bit_count, x)

    def forward(self, x):
        indices = x.argmax(-1)
        return torch.index_select(self.mapping, 0, indices)


class ClipQamDecoder(nn.Module):
    def __init__(self, bit_count, x=.5):
        super().__init__()
        self.mapping = clip_qam_mapping(bit_count, x)

    def forward(self, x):
        dist = torch.cdist(self.mapping, x)
        result = torch.argmin(dist, dim=0)
        return F.one_hot(result, num_classes=self.mapping.shape[0])


def ClipQamModulator(bit_count, noise):
    encoder = ClipQamEncoder(bit_count)
    decoder = ClipQamDecoder(bit_count)
    return Modulator(encoder, decoder, noise)
