import librosa 
import librosa.display
import librosa.beat
import sounddevice as sd 
import warnings
import numpy as np
import matplotlib.pyplot as plt

musicsfolder = "musics/"

def extract_features(folder):
    y, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr = sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y, sr = sr)
    f0 = librosa.yin(y=y,sr=sr)
    rms = librosa.feature.rms(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y, sr = sr)
    tempo = librosa.feature.tempo(y=y, sr = sr)
    return mfcc,spectral_centroid,spectral_bandwidth,spectral_contrast,spectral_flatness,f0,rms,zero_crossing_rate,tempo

if __name__== "__main__":
    extract_features(musicsfolder)