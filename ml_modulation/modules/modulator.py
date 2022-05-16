import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from .noise import AwgnNoise, ClippingNoise


class Modulator(pl.LightningModule):
    def __init__(self, encoder, decoder, noise):
        super().__init__()
        self.encoder = encoder
        self.noise = noise
        self.decoder = decoder
        self.loss_function = nn.CrossEntropyLoss()
        self.symbol_error_rate = torchmetrics.Accuracy()

    def forward(self, x):
        encoded = self.encoder(x)
        noised = self.noise(encoded)
        return self.decoder(noised)

    @torch.no_grad()
    def measure_error(self, dataloader, repeat=1000):
        self.eval()
        correct = 0
        total = 0
        for _ in range(repeat):
            for batch in dataloader:
                decoded = self(batch)
                prediction = decoded.argmax(-1)
                true_classes = batch.argmax(-1)

                correct += (prediction == true_classes).detach().cpu().sum().item()
                total += batch.shape[0]
        return 1 - correct / total

    @torch.no_grad()
    def test_snrs(self, dataloader, snrs, noise_name):
        result = {}
        default_noise = self.noise
        for snr in snrs:
            noise = AwgnNoise(snr) if noise_name == 'awgn' else ClippingNoise(snr)
            self.noise = noise
            error = self.measure_error(dataloader)
            result[snr] = error
        self.noise = default_noise
        return result

    def training_step(self, batch, batch_idx):
        decoded = self(batch)
        prediction = decoded.argmax(-1)
        true_classes = batch.argmax(-1)
        loss = self.loss_function(decoded, batch)
        ser = self.symbol_error_rate(prediction, true_classes)
        self.log('ser', ser, on_epoch=True, on_step=False, prog_bar=True)
        self.log('loss', loss, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        return optimizer


# class ModulatorAutoencoder(pl.LightningModule):
#     def __init__(self, class_count, encoding_shape, noise):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(class_count, 4 * class_count),
#             nn.ReLU(),
#             nn.Linear(4 * class_count, encoding_shape),
#             EntropyNormalization()
#         )
#         self.noise = noise
#         self.decoder = nn.Sequential(
#             nn.Linear(encoding_shape, 4 * class_count),
#             nn.ReLU(),
#             nn.Linear(4 * class_count, class_count),
#         )
#         self.loss_function = nn.CrossEntropyLoss()
#         self.symbol_error_rate = torchmetrics.Accuracy()
#
#     def forward(self, x):
#         encoded = self.encoder(x)
#         noised = self.noise(encoded)
#         return self.decoder(noised)
#
#     def measure_error(self, dataloader, repeat=1000):
#         correct = 0
#         total = 0
#         for _ in range(repeat):
#             for batch in dataloader:
#                 decoded = self(batch)
#                 prediction = decoded.argmax(-1)
#                 true_classes = batch.argmax(-1)
#
#                 correct += (prediction == true_classes).detach().cpu().sum().item()
#                 total += batch.shape[0]
#         return 1 - correct / total
#
#     def training_step(self, batch, batch_idx):
#         decoded = self(batch)
#         prediction = decoded.argmax(-1)
#         true_classes = batch.argmax(-1)
#         loss = self.loss_function(decoded, batch)
#         ser = self.symbol_error_rate(prediction, true_classes)
#         self.log('ser', ser, on_epoch=True, on_step=False, prog_bar=True)
#         self.log('loss', loss, on_epoch=True, on_step=False)
#         return loss
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
#         return optimizer
