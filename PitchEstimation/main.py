import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def read_signal_from_file(path):
    sample_rate, signal = wavfile.read(path)
    return sample_rate, signal


def auto_correlation_function(window):
    window_size = len(window)
    result = []
    for i in range(window_size):
        sigma = 0
        for j in range(window_size - i):
            sigma += window[j] * window[j + i]
        result.append(sigma / window_size)
    return result


def plot_signal(file_path):
    fig = plt.figure()
    axs = fig.subplots(nrows=2)
    sample_rate, signal = read_signal_from_file(file_path)
    # window size
    window_size = 1024
    # normalize signal
    normalized_signal = signal / np.amax(signal)
    x = np.arange(0, signal.shape[0])

    axs[1].plot(x, normalized_signal)

    auto_correlation_points = []
    start = 0
    end = window_size
    while end < signal.shape[0]:
        temp_window = np.float64(signal[start:end])
        for value in auto_correlation_function(temp_window):
            auto_correlation_points.append(value)
        start += window_size
        end += window_size
    for value in auto_correlation_function(np.float64(signal[start : signal.shape[0]])):
        auto_correlation_points.append(value)

    axs[0].plot(x, auto_correlation_points)


if __name__ == "__main__":

    plot_signal(file_path="PitchEstimation/TinHieuHuanLuyen/studio_F2.wav")
    plot_signal(file_path="PitchEstimation/TinHieuHuanLuyen/phone_F2.wav")

    plt.show()
