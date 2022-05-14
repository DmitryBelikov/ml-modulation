import matplotlib.pyplot as plt
import re
import torch

from .modules.noise import ClippingNoise, AwgnNoise
from .modules.qam import QamModulator
from .modules.autoencoder import NNModulator


def load_modulator(path):
    filename = path.name
    class2bites = {
        4: 2,
        16: 4,
        64: 6,
        256: 8,
        1024: 10,
        4096: 12
    }
    expression = r'^(?P<name>[a-zA-Z]+)_(?P<noise>[a-z]+)_(?P<class_count>\d+)_(?P<snr>[\d|.]+)_(?P<epochs>\d+)\.pt$'
    prog = re.compile(expression)
    match = prog.fullmatch(filename)
    if match is None:
        raise ValueError(f'Unable to match filename {filename}')
    result = match.groupdict()
    snr = float(result['snr'])
    noise_name = result['noise']
    noise = AwgnNoise(snr) if noise_name == 'awgn' else ClippingNoise(snr)
    class_count = int(result['class_count'])
    bit_count = class2bites[class_count]
    name = result['name']
    modulator = QamModulator(bit_count, noise) if name == 'qam' else NNModulator(bit_count, noise)
    modulator.load_state_dict(torch.load(path))
    modulator.eval()
    return modulator


def plot_constellation(modulator, dataloader, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)

    batch = next(iter(dataloader))
    points = modulator.encoder(batch)

    for point in points:
        ax.scatter(point[0], point[1], c='r')

