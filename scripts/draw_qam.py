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
    plt.xlabel('SNR [dB]')
    plt.ylabel('Bit error rate')

    # ax = plt.axis()
    # plt.axis((ax[0], ax[1], ax[3], ax[2]))
    plt.savefig('images/qam_ber.png')


def get_error_rates():
    qam4 = QAM(2)
    qam16 = QAM(4)
    qam64 = QAM(6)
    qam256 = QAM(8)
    message = np.random.choice([0, 1], size=48 * 1024)
    names = ['qam4', 'qam16', 'qam64', 'qam256']
    modulators = [qam4, qam16, qam64, qam256]
    encodings = [modulator.encode(message) for modulator in modulators]
    error_rates = {name: [] for name in names}
    for snr in np.arange(0., 50., 1.):
        n0 = 1 / 10 ** (snr / 10)
        for name, encoded, modulator in zip(names, encodings, modulators):
            noised = encoded + np.random.normal(0., np.sqrt(n0), encoded.shape)
            decoded = modulator.decode(noised)
            error_count = (decoded != message).sum()
            error_rate = error_count / (len(message) / modulator.bit_count)
            error_rates[name].append((snr, error_rate))
    return error_rates


if __name__ == '__main__':
    plot_info = get_error_rates()
    draw_plots(plot_info)
