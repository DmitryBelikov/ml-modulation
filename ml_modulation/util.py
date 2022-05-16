import matplotlib.pyplot as plt
import re

import numpy as np
from scipy.special import erfc

import torch
from torch.utils.data import DataLoader

from .datasets.modulator import ModulatorDataset
from .modules.noise import ClippingNoise, AwgnNoise
from .modules.qam import QamModulator
from .modules.clip_qam import ClipQamModulator
from .modules.autoencoder import NNModulator


def load_modulator(path, return_meta=False):
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
    if return_meta:
        return modulator, {
            'snr': snr,
            'noise': noise,
            'noise_name': noise_name,
            'class_count': class_count,
            'bit_count': bit_count,
            'name': name,
            'epochs': int(result['epochs'])
        }
    return modulator


@torch.no_grad()
def plot_constellation(modulator, dataloader, params=None, ax=None, size=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)

    batch = next(iter(dataloader))
    points = modulator.encoder(batch)
    if params is not None:
        ax.set_title(f'Noise: {params["noise_name"]}, M = {params["class_count"]}, SNR = {params["snr"]}')
    for point in points:
        if size:
            ax.scatter(point[0], point[1], c='r', s=size)
        else:
            ax.scatter(point[0], point[1], c='r')


def process_constellations(model_storage, cons_path):
    cons_path.mkdir(exist_ok=True)
    for file in model_storage.glob('*'):
        modulator, meta = load_modulator(file, return_meta=True)
        if meta['epochs'] != 50000:
            continue
        noise_name = meta['noise_name']
        snr = meta['snr']
        class_count = meta['class_count']
        bit_count = meta['bit_count']
        filename = f'{noise_name}_{class_count}_{snr}.png'
        figure, ax = plt.subplots(1, 1)
        dataset = ModulatorDataset(bit_count)
        dataloader = DataLoader(dataset, batch_size=class_count)
        size = {
            4: None,
            6: 15,
            8: 10
        }[bit_count]
        params = {
            'class_count': class_count,
            'snr': snr,
            'noise_name': noise_name
        }
        plot_constellation(modulator, dataloader, params, ax=ax, size=size)
        figure.savefig(cons_path / filename)


def Q(x):
    return erfc(x / np.sqrt(2)) / 2


def p_error(snr, q=16):
    sigma = 1 / 2 / (10 ** (snr / 10))
    n0 = 2 * sigma
    return 1 - (1 - 2 * Q(np.sqrt(3 / n0 / (q - 1)))) ** 2


@torch.no_grad()
def plot_awgn_theory(modulator, snrs, bit_count, left, right, step=0.05, ax=None):
    class_count = 2 ** bit_count
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.set_yscale('log')
    ax.set_ylabel('BLER')
    ax.set_xlabel('SNR, dB')
    theory_errors = [p_error(snr, class_count) for snr in np.arange(left, right, step)]
    ax.plot(np.arange(left, right, step), theory_errors, label=f'QAM{class_count} error')
    error_rates = []
    dataset = ModulatorDataset(bit_count)
    dataloader = DataLoader(dataset, batch_size=class_count)
    for snr in snrs:
        modulator.noise = AwgnNoise(snr)
        error_rates.append(modulator.measure_error(dataloader))
    ax.plot(snrs, error_rates, label='Autoencoder error')
    ax.legend()


def process_all_theory(model_storage, theory_dir):
    theory_dir.mkdir(exist_ok=True)
    ranges = {
        4: (11, 19),
        6: (16, 25),
        8: (23, 31)
    }
    main_snr = {
        4: 16.,
        6: 20.,
        8: 29.
    }
    for bit_count in [4, 6, 8]:
        figure, ax = plt.subplots(1, 1)
        class_count = 2 ** bit_count
        snr = main_snr[bit_count]
        modulator = load_modulator(model_storage / f'ae_awgn_{class_count}_{snr}_50000.pt')
        left, right = ranges[bit_count]
        plot_awgn_theory(modulator, np.arange(left + 1, right - 1, .5), bit_count, left, right, ax=ax)
        figure.savefig(theory_dir / f'theory_{bit_count}.png')


@torch.no_grad()
def clipping_pairwise(model_storage, save_dir, draw_qam=True):
    save_dir.mkdir(exist_ok=True)
    snrs = {
        4: [14., 15., 16., 17., 18., 19., 20.],
        6: [17., 18., 19., 20., 21., 22., 23., 24.],
        8: [24., 25., 26., 27., 28., 29., 30.],
    }
    for bit_count in [4, 6, 8]:
        class_count = 2 ** bit_count
        qam = QamModulator(bit_count, ClippingNoise(10))
        clip_qam = ClipQamModulator(bit_count, ClippingNoise(10))

        qam_errors = []
        clip_qam_errors = []
        ae_errors = []

        dataset = ModulatorDataset(bit_count)
        dataloader = DataLoader(dataset, batch_size=class_count)
        for snr in snrs[bit_count]:
            filename = f'ae_clipping_{class_count}_{snr}_50000.pt'
            path = model_storage / filename
            autoencoder = load_modulator(path)
            qam.noise = ClippingNoise(snr)
            clip_qam.noise = ClippingNoise(snr)

            ae_errors.append(autoencoder.measure_error(dataloader))
            qam_errors.append(qam.measure_error(dataloader))
            clip_qam_errors.append(clip_qam.measure_error(dataloader))

        figure, ax = plt.subplots(1, 1)
        ax.set_yscale('log')
        ax.set_ylabel('BLER')
        ax.set_xlabel('SNR, dB')
        ax.plot(snrs[bit_count], ae_errors, label='Autoencoder error')
        if draw_qam:
            ax.plot(snrs[bit_count], qam_errors, label=f'QAM{class_count} error')
        ax.plot(snrs[bit_count], clip_qam_errors, label=f'ClipQAM{class_count} error')
        ax.legend()
        addition = '' if draw_qam else 'no_qam_'
        figure.savefig(save_dir / f'clipping_{addition}{bit_count}.png')
