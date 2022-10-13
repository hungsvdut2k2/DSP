from curses import window
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


if __name__ == "__main__":
    sample_rate, signal = read_signal_from_file(
        "PitchEstimation/TinHieuHuanLuyen/phone_M2.wav"
    )
    # window size
    window_size = 1024
    # normalize signal
    normalized_signal = signal / np.amax(signal)
    x = np.arange(0, signal.shape[0])
    plt.figure()
    plt.plot(x, normalized_signal)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")

    auto_correlation_points = []
    start = 0
    end = 1024
    while end < signal.shape[0]:
        temp_window = np.float64(signal[start:end])
        for value in auto_correlation_function(temp_window):
            auto_correlation_points.append(value)
        start += 1024
        end += 1024
    for value in auto_correlation_function(np.float64(signal[start : signal.shape[0]])):
        auto_correlation_points.append(value)
    plt.figure()
    plt.plot(x, auto_correlation_points)

    plt.show()
