import torch
from torch.utils.data import Dataset


class ModulatorDataset(Dataset):
    def __init__(self, bit_count):
        super().__init__()
        self.bit_count = bit_count

    def __len__(self):
        return 2 ** self.bit_count

    def __getitem__(self, idx):
        result = torch.zeros(2 ** self.bit_count)
        result[idx] = 1
        return result
