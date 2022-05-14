import numpy as np

from .qam import QAM


class ClippedQAM(QAM):
    def __init__(self, bit_count, x_clip):
        super().__init__(bit_count)
        max_x = np.abs(self.digit_mapping[:, 0]).max()
        self.digit_mapping[:, 0] *= x_clip / max_x
        x_energy = (self.digit_mapping[:, 0] ** 2).sum()
        cur_y_energy = (self.digit_mapping[:, 1] ** 2).sum()
        result_y_energy = 2 ** bit_count - x_energy
        self.digit_mapping[:, 1] *= np.sqrt(result_y_energy / cur_y_energy)
