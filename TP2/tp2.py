import librosa 
import librosa.display
import librosa.beat
import sounddevice as sd 
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

musicsfolder = "musics/"

def extract_features(folder):
    feature = np.zeros()
    y, sr = librosa.load(folder, sr=None)
    feature.append(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
    feature.append(librosa.feature.spectral_centroid(y=y, sr=sr))
    feature.append(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    feature.append(librosa.feature.spectral_contrast(y=y, sr = sr))
    feature.append(librosa.feature.spectral_flatness(y=y))
    feature.append(librosa.yin(y=y,sr=sr,fmin=65,fmax=2093))
    feature.append(librosa.feature.rms(y=y))
    feature.append(librosa.feature.zero_crossing_rate(y=y))
    feature.append(librosa.feature.tempo(y=y, sr = sr))

    return feature
def statistics(feature):
    stats = np.zeros()
    stats.append()
    stats.append()
    stats.append()
    stats.append()
    stats.append()
    stats.append()
    stats.append()
    return stats
def save_to_csv(features_list, output_file="audio_features.csv"):
    df = pd.DataFrame(features_list)
    df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

if __name__== "__main__":
    music_test = 10
    musics = os.listdir(musicsfolder)
    features = []

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
        features.append(feature)
        music_test-=1

    save_to_csv(features)