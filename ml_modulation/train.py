import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from datasets.modulator import ModulatorDataset

from modules.autoencoder import NNModulator
from modules.noise import AwgnNoise, ClippingNoise


def train(params, snr, model=None):
    class_count = 2 ** params.bits
    dataset = ModulatorDataset(params.bits)
    dataloader = DataLoader(dataset, batch_size=class_count, shuffle=True, pin_memory=True)
    noise = AwgnNoise(snr) if params.noise == 'awgn' else ClippingNoise(snr, 0.5, 1000.)
    if model:
        model.noise = noise
    else:
        model = NNModulator(params.bits, noise)
    logger = CSVLogger(save_dir=Path(params.log_dir))
    trainer = pl.Trainer(max_epochs=params.epochs, logger=logger, gpus=[params.gpu])
    trainer.fit(model, train_dataloaders=dataloader)

    filename = f'ae_{params.noise}_{class_count}_{snr}_{params.epochs}.pt'
    root_dir = Path(params.model_dir)
    root_dir.mkdir(exist_ok=True)
    model_path = root_dir / filename

    torch.save(model.state_dict(), model_path)

    return model


def train_models(params):
    snrs = sorted(params.snr)
    model = None
    for snr in snrs:
        model = train(params, snr, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bits', type=int)
    parser.add_argument('--snr', '--snrs', nargs='+', type=float)
    parser.add_argument('--noise', type=str, choices=['awgn', 'clipping'])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--model_dir', type=str, default='new_model_storage/')
    args = parser.parse_args()
    train_models(args)
