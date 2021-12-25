import itertools
import matplotlib.pyplot as plt
import numpy as np


def constellation_diagram(modulator, output_file):
    figure, axes = plt.subplots()
    bit_count = modulator.bit_count
    combinations = itertools.product([0, 1], repeat=bit_count)
    xs = []
    ys = []
    labels = []
    for bit_sequence in combinations:
        input_data = np.array(bit_sequence)
        [[x, y]] = modulator.encode(input_data)
        xs.append(x)
        ys.append(y)
        labels.append(''.join(map(str, bit_sequence)))
    axes.scatter(xs, ys, marker='o', color='r')
    for x, y, label in zip(xs, ys, labels):
        axes.annotate(label, (x, y))
    figure.savefig(output_file, bbox_inches='tight')
