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
    voice = []
    ste_values = []
    result = []
    sample_rate, signal = wavfile.read(file_path)
    frame_size = int(frame_size * sample_rate)
    start = 0
    end = frame_size
    while end < signal.shape[0]:
        temp_list = []
        temp_frame = np.float64(signal[start:end])
        value = eval(f"{attribute_function}(temp_frame)")
        ste_values.append(value)
        temp_list.append(value)
        temp_list.append(temp_frame)
        voice.append(temp_list)
        start += frame_size
        end += frame_size
    for sublist in voice:
        sublist[0] /= max(ste_values)
    for sublist in voice:
        if sublist[0] >= 0.01:
            for value in sublist[1]:
                result.append(value)
    return result


def get_middle_frame(frame):
    frame_size = len(frame) / 3
    start = int(frame_size)
    end = int(2 * frame_size)
    return frame[start:end]


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
        result.append(librosa.stft(temp_frame, n_fft))
        start += frame_size
        end += frame_size
    if end < middle_frames.shape[0]:
        temp_frame = middle_frames[end : middle_frames.shape[0]]
        result.append(librosa.stft(temp_frame, n_fft))
    return np.mean(result)


def features_extraction(directory):
    all_directories = iterator_for_all_sub_directories(directory)
    result = {"a.wav": [], "e.wav": [], "i.wav": [], "o.wav": [], "u.wav": []}
    for directory in all_directories:
        for value in all_directories[directory]:
            sample_rate, signal = wavfile.read(value)
            frame_size = int(0.03 * sample_rate)
            result[value.split("/")[-1]].append(fft(signal, frame_size, 512))
    return result


def euclidean_distance(first_vector, second_vector):
    return np.sum((first_vector - second_vector) ** 2)


if __name__ == "__main__":
    file_path = "/home/viethung/DSP/VowelRegconition/NguyenAmHuanLuyen-16k/01MDA/a.wav"
    result = voice_segment(file_path, 0.03, 0.01, "short_term_energy")
    print(len(result))
    other_result = get_middle_frame(result)
    print(len(other_result))
