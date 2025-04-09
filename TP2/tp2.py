import librosa 
import librosa.display
import librosa.beat
import sounddevice as sd 
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import os
from types import NoneType

musicsfolder = "Music/"

def extract_features(folder):
    feature = []
    y, sr = librosa.load(folder, sr=None)

    mfc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for line in mfc:
        stats = statistics(line)
        feature.extend(stats)
    
    spec_centr = librosa.feature.spectral_centroid(y=y, sr=sr)
    stats = statistics(spec_centr)
    feature.extend(stats)

    spec_band = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    stats = statistics(spec_band)
    feature.extend(stats)

    spec_cont = librosa.feature.spectral_contrast(y=y, sr = sr)
    stats = statistics(spec_cont)
    feature.extend(stats)

    spec_flat = librosa.feature.spectral_flatness(y=y)
    stats = statistics(spec_flat)
    feature.extend(stats)

    spec_roll = librosa.feature.spectral_rolloff(y=y, sr = sr)
    stats = statistics(spec_roll)
    feature.extend(stats)

    fzero = librosa.yin(y=y,sr=sr,fmin=65,fmax=2093)
    stats = statistics(fzero)
    feature.extend(stats)

    rms = librosa.feature.rms(y=y)
    stats = statistics(rms)
    feature.extend(stats)

    zcr = librosa.feature.zero_crossing_rate(y=y)
    stats = statistics(zcr)
    feature.extend(stats)
    
    tempo = librosa.feature.tempo(y=y, sr = sr)
    feature.append(tempo[0])

    return feature


def statistics(feature):
    feature = np.float64(feature).transpose()
    stats = []
    stats.append(np.mean(feature))
    stats.append(np.std(feature))

    temp = scipy.stats.skew(feature)
    if type(temp) == np.ndarray:
        temp = temp[0]
    stats.append(temp)

    temp = scipy.stats.kurtosis(feature)
    if type(temp) == np.ndarray:
        temp = temp[0]
    stats.append(temp)

    stats.append(np.median(feature))
    stats.append(np.max(feature))
    stats.append(np.min(feature))

    return stats


def save_to_csv(features_list, output_file="audio_features.csv"):
    np.savetxt(output_file, features_list, "%.6f", delimiter=", ")
    print(f"Features saved to {output_file}")


def normalize(value ,col_min, col_max):
    new_value = (value - col_min) / (col_max - col_min)
    return new_value


if __name__== "__main__":
    music_test = 900
    musics = os.listdir(musicsfolder)
    features = None

    # for music in musics:
    #     print(f"Processing: {music}")
    #     path = os.path.join(musicsfolder, music)
    #     feature = extract_features(path)
    #     features.append(feature)

    for music in musics:
        if music_test == 0:
            break

        print(f"Processing: {music}")
        path = os.path.join(musicsfolder, music)

        feature = extract_features(path)
        print(f"Número de features: {len(feature)}")
        
        if type(features) == NoneType:
            features = np.array(feature)
        else:
            features = np.vstack((features, feature))
        
        music_test-=1

    maximos = []
    minimos = []

    for column in range (features.shape[1]):
        col_min = np.min(features[:,column])
        minimos.append(col_min)

        col_max = np.max(features[:,column])
        maximos.append(col_max)

        if col_max == col_min:
            features[:, column] = 1
            continue

        for line in range (features.shape[0]):
            value = features[line , column]
            new_value = normalize(value, col_min, col_max)
            features[line , column] = new_value

    features = np.vstack((minimos, maximos, features))
    save_to_csv(features)
