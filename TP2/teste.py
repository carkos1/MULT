import librosa 
import librosa.display
import librosa.beat
import sounddevice as sd 
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from mutagen.mp3 import MP3  # For metadata extraction
from sklearn.preprocessing import OneHotEncoder
import joblib  # For parallel processing

musicsfolder = "musics/"

def extract_features(folder):
    y, sr = librosa.load(folder, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr = sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    f0 = librosa.yin(y=y,sr=sr,fmin=65,fmax=2093)
    rms = librosa.feature.rms(y=y)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    tempo = librosa.feature.tempo(y=y, sr = sr)
    return mfcc,spectral_centroid,spectral_bandwidth,spectral_contrast,spectral_flatness,f0,rms,zero_crossing_rate,tempo

def save_to_csv(features_list, output_file="audio_features.csv"):
    df = pd.DataFrame(features_list)
    df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

if __name__== "__main__":

    save_path = "features.csv"

    musics = os.listdir(musicsfolder)
    

    for music in musics:
        print(f"Processing: {music}")
        path = os.path.join(musicsfolder, music)
        mfcc,spectral_centroid,spectral_bandwidth,spectral_contrast,spectral_flatness,f0,rms,zero_crossing_rate,tempo = extract_features(path)
    save_to_csv()