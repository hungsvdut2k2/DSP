from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os

SAMPLE_DIR = r"/home/viethung/DSP/VowelRegconition/NguyenAmHuanLuyen-16k"


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


def iterator_for_all_sub_directories(directory):
    files_path = {"a.wav": [], "e.wav": [], "i.wav": [], "o.wav": [], "u.wav": []}
    for sub_directory in os.scandir(directory):
        for file in os.scandir(sub_directory):
            splitted_file_path = file.path.split("/")
            files_path[splitted_file_path[-1]].append(file.path)
    return files_path


def fft(middle_frames, frame_size, n_fft):
    result = []
    start = 0
    end = frame_size
    while end < middle_frames.shape[0]:
        temp_frame = middle_frames[start:end]
        result.append(librosa.sfft(temp_frame, n_fft))
        start += frame_size
        end += frame_size
    if end < middle_frames.shape[0]:
        temp_frame = middle_frames[end : middle_frames.shape[0]]
        result.append(librosa.sfft(temp_frame, n_fft))
    return np.mean(result)


if __name__ == "__main__":

    all_directories = iterator_for_all_sub_directories(SAMPLE_DIR)
    for directory in all_directories:
        for value in all_directories[directory]:
            print(value)
