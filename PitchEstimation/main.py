import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os


SAMPLE_DIR = r"/home/viethung/DSP/PitchEstimation/TinHieuKiemThu"


def read_signal_from_file(path):
    sample_rate, signal = wavfile.read(path)
    return sample_rate, signal


def get_label(path):
    frames = {"sil": [], "v": [], "uv": []}
    with open(path) as f:
        content = f.readlines()
        for line in content:
            focused_content = line.split()
            if focused_content[0] != "F0mean" and focused_content[0] != "F0std":
                frames[focused_content[2]].append(
                    (focused_content[0], focused_content[1])
                )
    return frames


def auto_correlation_function(window):
    window_size = len(window)
    result = []
    for i in range(window_size):
        sigma = 0
        for j in range(window_size - i):
            sigma += window[j] * window[j + i]
        result.append(sigma / window_size)
    return result


def find_peaks(window):
    max_peak = 0
    max_index = 0
    for i in range(2, len(window) - 1):
        if (
            window[i] > window[i - 1]
            and window[i] > window[i + 1]
            and window[i] > max_peak
        ):
            max_peak = window[i]
            max_index = i
    return max_peak, max_index


def plot_signal(file_path):
    fs, signal = read_signal_from_file(file_path)

    voice_start = 0
    voice_end = 0
    unvoice_start = 0
    unvoice_end = 0
    if file_path.split(".")[0] == "phone_F1":
        voice_start = 0.53
        voice_end += 0.56
        unvoice_start = 1.14
        unvoice_end += 1.17
    elif file_path.split(".")[0] == "phone_M1":
        voice_start = 0.46
        voice_end += 0.49
        unvoice_start = 1.39
        unvoice_end += 1.42
    elif file_path.split(".")[0] == "studio_F1":
        unvoice_start = 0.68
        unvoice_end = 0.7
        voice_start = 1.10
        voice_end = 1.13
    else:
        unvoice_start = 0.87
        unvoice_end += 0.9
        voice_start = 0.94
        voice_end += 0.97
    window_size = int(0.03 * fs)
    x = np.arange(0, signal.shape[0])
    threshold = 0.3
    fig = plt.figure()
    axs = fig.subplots(nrows=4)
    # plot the input signal
    axs[3].plot(x, signal)
    axs[3].set_xlabel("Sample Index")
    axs[3].set_ylabel("Amplitude")

    # auto correlation function
    f0 = np.zeros(x.shape[0])
    start = 0
    end = window_size
    points = []
    voice = []
    unvoice = []
    while end < signal.shape[0]:
        temp_window = np.float64(signal[start:end])
        acf_value = auto_correlation_function(temp_window)
        # normalize value
        max_value = np.max(acf_value)
        if max_value != 0:
            acf_value /= max_value
        peak, index = find_peaks(acf_value)
        if peak > threshold:
            if (fs / index) >= 70 and (fs / index) < 450:
                f0[end] = fs / index
                points.append(fs / index)
        start += window_size
        end += window_size
    f0[np.where(f0 == 0)] = np.nan
    mean_of_points = np.mean(np.array(points))
    for value in f0:
        if np.isnan(value) != True:
            if value < 0.25 * mean_of_points or value > 4 * mean_of_points:
                value = np.nan
                points.remove(value)
    res_mean = np.mean(np.array(points))
    res_std = np.std(np.array(points))
    # plot pitch contour
    axs[2].scatter(x, f0)
    # plot voice
    temp_window = signal[int(voice_start * fs) : int(voice_end * fs)]
    acf_value = auto_correlation_function(temp_window)
    for value in acf_value:
        voice.append(value)
    axs[1].plot(np.arange(len(voice)), voice)
    # plot unvoice
    second_temp_window = signal[int(unvoice_start * fs) : int(unvoice_end * fs)]
    second_acf_value = auto_correlation_function(second_temp_window)
    for value in second_acf_value:
        unvoice.append(value)
    axs[0].plot(np.arange(len(unvoice)), unvoice)


if __name__ == "__main__":
    wave_names = os.listdir(SAMPLE_DIR)
    wave_names = [item for item in wave_names if item.endswith(".wav")]
    for index, wave_name in enumerate(wave_names):
        file_name = wave_name[: wave_name.rfind(".")]
        print(">> Processing on " + wave_name)
        wave_path = os.path.join(SAMPLE_DIR, wave_name)
        plot_signal(wave_path)
    plt.show()
