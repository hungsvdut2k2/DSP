import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

SAMPLE_DIR = r"/home/viethung/DSP/VowelRegconition/NguyenAmHuanLuyen-16k"


def short_term_energy(frame):
    return np.sum(frame**2) / frame.shape[0]


def iterator_for_all_sub_directories(directory):
    files_path = {"a.wav": [], "e.wav": [], "i.wav": [], "o.wav": [], "u.wav": []}
    for sub_directory in os.scandir(directory):
        for file in os.scandir(sub_directory):
            splitted_file_path = file.path.split("/")
            files_path[splitted_file_path[-1]].append(file.path)
    return files_path


def mfcc(middle_frame, sample_rate):
    result = []
    frame_size = int(sample_rate * 0.03)
    start = 0
    end = frame_size
    while end < frame_size:
        temp_frame = middle_frame[start:end]
        result.append(librosa.feature.mfcc(y=temp_frame, sr=sample_rate, n_mfcc=13))
        start += frame_size
        end += frame_size
    return result


if __name__ == "__main__":
    sample_rate, signal = wavfile.read(
        "/home/viethung/DSP/VowelRegconition/NguyenAmHuanLuyen-16k/01MDA/a.wav"
    )
    print(librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13))
