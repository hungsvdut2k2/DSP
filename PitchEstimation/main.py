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
        "PitchEstimation/TinHieuHuanLuyen/phone_F2.wav"
    )
