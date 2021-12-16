import numpy as np
import matplotlib.pyplot as plt

from ml_modulation.modulators.qam import QAM


def draw_plots(error_rates):
    for name, data in error_rates.items():
        data = sorted(data)
        xs = [x for x, _ in data]
        ys = [y for _, y in data]
        plt.scatter(xs, ys, s=16)
        plt.plot(xs, ys, ls='--', label=name)

    plt.legend()
    plt.yscale('log')
    plt.xlabel('E / N0')
    plt.ylabel('Bit error rate')

    # ax = plt.axis()
    # plt.axis((ax[0], ax[1], ax[3], ax[2]))
    plt.savefig('images/qam_ser.png')


def get_error_rates():
    qam4 = QAM(2)
    qam16 = QAM(4)
    qam64 = QAM(6)
    qam256 = QAM(8)
    message = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
                       * 3 * 1024 * 16)
    names = ['qam4', 'qam16', 'qam64', 'qam256']
    modulators = [qam4, qam16, qam64, qam256]
    encodings = [modulator.encode(message) for modulator in modulators]
    energies = [2 / 3 * (modulator.bit_count + 1) / (modulator.bit_count - 1)
                for modulator in modulators]
    error_rates = {name: [] for name in names}
    for edivN0 in np.arange(0.2, 16, .3):
        for name, energy, encoded, modulator in zip(names, energies, encodings,
                                                    modulators):
            n0 = energy / edivN0
            noised = encoded + \
                     np.random.normal(0., np.sqrt(n0 / 2), encoded.shape)
            decoded = modulator.decode(noised)
            error_count = \
                ((decoded != message).
                 reshape((-1, modulator.bit_count)).
                 sum(axis=1) != 0).sum()
            error_rate = error_count / (len(message) / modulator.bit_count)
            error_rates[name].append((energy / n0, error_rate))
    return error_rates


if __name__ == '__main__':
    plot_info = get_error_rates()
    draw_plots(plot_info)
