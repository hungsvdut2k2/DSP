from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import librosa


def short_term_energy(frame):
    return np.sum(frame**2)


def mean_average(frame):
    return np.sum(abs(frame))


def zero_crossing_rate(frame):
    return np.sum(np.abs(np.sign(frame[:-1]) - np.sign(frame[1:])))


def voice_segment(file_path, frame_size, threshold, attribute_function):
    sample_rate, signal = wavfile.read(file_path)
    frame_size = int(frame_size * sample_rate)
    start = 0
    end = frame_size
    value = []
    while end < signal.shape[0]:
        temp_frame = np.float64(signal[start:end])
        value.append(eval(f"{attribute_function}(temp_frame)"))
        start += frame_size
        end += frame_size
    return value


if __name__ == "__main__":

    file_path = "/home/viethung/DSP/VowelRegconition/NguyenAmHuanLuyen-16k/01MDA/a.wav"
    print(voice_segment(file_path, 0.03, 0, "zero_crossing_rate"))
