import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def read_signal_from_file(path):
    sample_rate, signal = wavfile.read(path)
    return sample_rate, signal


if __name__ == "__main__":
    sample_rate, signal = read_signal_from_file(
        "/home/viethung/DSP/TinHieuHuanLuyen/phone_F2.wav"
    )
    # plot input signal
    x = np.arange(0, signal.shape[0])
    plt.figure()
    plt.plot(x, signal)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.xticks(np.arange(0, 6) * sample_rate, labels=[0, 1, 2, 3, 4, 5])

    normalized_signal = signal / np.amax(signal)
    x = np.arange(0, signal.shape[0])
    plt.figure()
    plt.plot(x, normalized_signal)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.xticks(np.arange(0, 6) * sample_rate, labels=[0, 1, 2, 3, 4, 5])

    plt.show()
