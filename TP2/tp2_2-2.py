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

window_size = 2048
hop_length = 512

musicsfolder = "musics/"

def spectralCentroid(y, sr):
    zp = hop_length - len(y) % hop_length
    
    np.pad(y, (0, zp), 'constant')
    
    n_frames = (len(y) - window_size) // hop_length + 1
    hann = np.hanning(window_size)
    
    sc = []
    
    for i in range(n_frames):
        start = i * hop_length
        end = start + window_size
        janela = y[start:end]
        janela_hann = janela * hann
        
        x = scipy.fft.rfft(janela_hann)
        mag_x = np.abs(x)
        
        frequency = scipy.fft.rfftfreq(window_size, 1/sr)
        
        mag_total = np.sum(mag_x)
        if mag_total == 0:
            centroid = 0.0
        else:
            centroid = np.sum(mag_x * frequency) / mag_total
        sc.append(centroid)
        
    return sc
        
def scCompare(sc_manual, spec_centr):
    sc_manual = np.array(sc_manual)
    
    # Flatten librosa spectral centroid to 1D array
    if spec_centr.ndim > 1:
        spec_centr = spec_centr.flatten()
        
    # Ensure both arrays are the same length
    min_len = min(len(sc_manual), len(spec_centr))
    sc_manual = sc_manual[:min_len]
    spec_centr = spec_centr[:min_len]
    
    mse = np.mean((sc_manual - spec_centr) ** 2)
    rmse = np.sqrt(mse)
    
    coef = np.corrcoef(sc_manual, spec_centr)[0, 1]
    
    return rmse, coef
    

def save_to_csv(features_list, output_file="sc.csv"):
    np.savetxt(output_file, features_list, "%.6f", delimiter=", ")
    print(f"Features saved to {output_file}")
    
if __name__== "__main__":
    music_test = 900
    musics = os.listdir(musicsfolder)
    
    array_erros = []
    
    for music in musics:
        array_e = []
        if music_test == 0:
            break

        print(f"Processing: {music}")
        path = os.path.join(musicsfolder, music)
        y, sr = librosa.load(path, sr = 22050)
        
        sc_manual = spectralCentroid(y, sr)
        spec_centr = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        spec_centr = spec_centr[:, 2:]
        
        
        rmse, coef = scCompare(sc_manual, spec_centr)
        
        array_e.append(coef)
        array_e.append(rmse)
        
        array_erros.append(array_e)
        
        music_test-=1
        
    save_to_csv(array_erros)
