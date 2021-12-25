import numpy as np

from .exceptions import ModulationException, EncodingException
from .interfaces import Modulator


class QAM(Modulator):
    def __init__(self, bit_count):
        if bit_count % 2 != 0:
            raise ModulationException(
                f'QAM defined for even bit_counts only. '
                f'Got bit count={bit_count}')
        self.bit_count = bit_count
        self.values_count = 2 ** bit_count
        base = 2 ** (bit_count // 2)
        self.digit_mapping = np.zeros((self.values_count, 2))
        for value in range(self.values_count):
            first_digit = value // base
            second_digit = value % base
            first_signal = (1 - 2 * first_digit / (base - 1))
            second_signal = (1 - 2 * second_digit / (base - 1))
            self.digit_mapping[value] = np.array([first_signal, second_signal])
        total_energy = (self.digit_mapping ** 2).sum()
        self.digit_mapping /= np.sqrt(total_energy)

    def encode(self, bits):
        if len(bits) % self.bit_count != 0:
            raise EncodingException(
                'Length of bits array should be divisible '
                'by modulator bit count')
        result = []
        current_number = 0
        rest_bits = 0
        for bit in bits:
            current_number *= 2
            current_number += bit
            rest_bits += 1
            if rest_bits == self.bit_count:
                result.append(self.digit_mapping[current_number])
                rest_bits = 0
                current_number = 0
        return np.array(result)

    def decode(self, encoded):
        bit_lines = []
        for xy in encoded:
            closest_value = \
                np.argmin(np.linalg.norm(self.digit_mapping - xy, axis=1))
            current_bits = np.unpackbits(np.array(closest_value, dtype='ubyte'))
            bit_lines.append(current_bits[8 - self.bit_count:])
        return np.concatenate(bit_lines)
