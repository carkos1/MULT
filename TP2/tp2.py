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
import csv
from types import NoneType

musicsfolder = "Music/"
notNorm = "validação de resultados_TP2/notNormFM_All.csv"

def extract_features(folder):
    feature = [0] * 190
    sr = 22050

    y, fs = librosa.load(folder, sr= sr)
    index = 0

    mfc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for line in mfc:
        stats = statistics(line)
        feature[index : index + len(stats)] = stats
        index += len(stats)
     
    spec_centr = librosa.feature.spectral_centroid(y=y)
    stats = statistics(spec_centr.flatten())
    feature[index : index + len(stats)] = stats
    index += len(stats)

    spec_band = librosa.feature.spectral_bandwidth(y=y)
    stats = statistics(spec_band.flatten())
    feature[index : index + len(stats)] = stats
    index += len(stats)

    spec_cont = librosa.feature.spectral_contrast(y=y)
    for line in spec_cont:
        stats = statistics(line)
        feature[index : index + len(stats)] = stats
        index += len(stats)

    spec_flat = librosa.feature.spectral_flatness(y=y)
    stats = statistics(spec_flat.flatten())
    feature[index : index + len(stats)] = stats
    index += len(stats)

    spec_roll = librosa.feature.spectral_rolloff(y=y)
    stats = statistics(spec_roll.flatten())
    feature[index : index + len(stats)] = stats
    index += len(stats)

    fzero = librosa.yin(y=y, sr = sr ,fmin = 20, fmax = sr / 2)
    #f0[f0== fs/2] = 0
    stats = statistics(fzero)
    feature[index : index + len(stats)] = stats
    index += len(stats)

    rms = librosa.feature.rms(y=y)
    stats = statistics(rms.flatten())
    feature[index : index + len(stats)] = stats
    index += len(stats)

    zcr = librosa.feature.zero_crossing_rate(y=y)
    stats = statistics(zcr.flatten())
    feature[index : index + len(stats)] = stats
    index += len(stats)
    
    tempo = librosa.feature.tempo(y=y)
    feature[189] = tempo[0]

    return feature


def statistics(feature):
    stats = [0] * 7
    stats[0] = np.mean(feature)
    stats[1] = np.std(feature)

    temp = scipy.stats.skew(feature)
    if type(temp) == np.ndarray:
        temp = temp[0]
    stats[2] = temp

    temp = scipy.stats.kurtosis(feature)
    if type(temp) == np.ndarray:
        temp = temp[0]
    stats[3] = temp

    stats[4] = np.median(feature)
    stats[5] = np.max(feature)
    stats[6] = np.min(feature)

    return stats


def save_to_csv(features_list, output_file="audio_features.csv"):
    np.savetxt(output_file, features_list, "%.6f", delimiter=",")
    print(f"Features saved to {output_file}")




def normalize(value ,col_min, col_max):
    new_value = (value - col_min) / (col_max - col_min)
    return new_value


if __name__== "__main__":
    music_test = 900
    musics = os.listdir(musicsfolder)
    features = None

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

    save_to_csv(features)

    maximos = [0] * 190
    minimos = [0] * 190

    for column in range (features.shape[1]):
        col_min = np.min(features[:,column])
        minimos[column] = col_min

        col_max = np.max(features[:,column])
        maximos[column] = col_max

        if col_max == col_min:
            features[:, column] = 1
            continue

        for line in range (features.shape[0]):
            value = features[line , column]
            new_value = normalize(value, col_min, col_max)
            features[line , column] = new_value

    features = np.vstack((minimos, maximos, features))
    
