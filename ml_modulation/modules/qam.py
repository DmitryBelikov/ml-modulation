import torch
import torch.nn as nn
import torch.nn.functional as F

from .modulator import Modulator


def qam_mapping(bit_count):
    result_size = 2 ** bit_count
    base = 2 ** (bit_count // 2)
    mapping = [[0, 0] for _ in range(result_size)]
    for index in range(2 ** bit_count):
        first_digit = index // base
        second_digit = index % base
        first_signal = (1 - 2 * first_digit / (base - 1))
        second_signal = (1 - 2 * second_digit / (base - 1))

        mapping[index] = [first_signal, second_signal]
    result = torch.tensor(mapping)
    total_energy = (result ** 2).sum()
    result /= torch.sqrt(total_energy)
    result *= base
    return result


class QamEncoder(nn.Module):
    def __init__(self, bit_count):
        super().__init__()
        self.mapping = qam_mapping(bit_count)

    def forward(self, x):
        indices = x.argmax(-1)
        return torch.index_select(self.mapping, 0, indices)


class QamDecoder(nn.Module):
    def __init__(self, bit_count):
        super().__init__()
        self.mapping = qam_mapping(bit_count)

    def forward(self, x):
        dist = torch.cdist(self.mapping, x)
        result = torch.argmin(dist, dim=0)
        return F.one_hot(result, num_classes=self.mapping.shape[0])


def QamModulator(bit_count, noise):
    encoder = QamEncoder(bit_count)
    decoder = QamDecoder(bit_count)
    return Modulator(encoder, decoder, noise)
