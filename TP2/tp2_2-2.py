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

window_size = 2048
hop_length = 512

musicsfolder = "musics/"
ranking_output = "rankings.txt"

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
    
def euclidean_distance(cat1, cat2):
    return np.linalg.norm(cat1 - cat2)

def manhattan_distance(cat1, cat2):
    return np.sum(np.abs(cat1 - cat2))

#Já agora incluo a distância de Chebyshev
def chebyshev_distance(cat1, cat2):
    return max(np.abs(cat1 - cat2))

#E a distância de Minkowski :D
def minkowski_distance(cat1, cat2, p=3):
    return np.power(np.sum(np.abs(cat1 - cat2) ** p), 1/p)

def cosine_distance(cat1, cat2):
    return 1 - np.dot(cat1, cat2) / (np.linalg.norm(cat1) * np.linalg.norm(cat2))

def rank_euc(de, dm, dc):
    # Rank the distances
    de_ranked = np.argsort(de)
    dm_ranked = np.argsort(dm)
    dc_ranked = np.argsort(dc)

    return de_ranked, dm_ranked, dc_ranked

def save_to_csv(features_list, output_file):
    np.savetxt(output_file, features_list, "%.6f", delimiter=", ")
    print(f"Features saved to {output_file}")

if __name__== "__main__":
    music_test = 900
    musics = os.listdir(musicsfolder)
    dist_calc = 1
    dist_array = np.zeros((3, 900))

    fm_Q = pd.read_csv('validação de resultados_TP2/FM_Q.csv')

    fm_all = pd.read_csv('validação de resultados_TP2/FM_All.csv')

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
        
         # Calculate distances
         de = euclidean_distance(fm_Q.iloc[1, :].values, fm_all.iloc[dist_calc, :].values)
         dm = manhattan_distance(fm_Q.iloc[1, :].values, fm_all.iloc[dist_calc, :].values)
         dc = cosine_distance(fm_Q.iloc[1, :].values, fm_all.iloc[dist_calc, :].values)


         dist_array[0,dist_calc-1] = de
         dist_array[1,dist_calc-1] = dm
         dist_array[2,dist_calc-1] = dc

         dist_calc += 1
         print("Música num:", dist_calc-1)
         music_test-=1

    save_to_csv(array_erros, "sc.csv")

    de = dist_array[0]
    dm = dist_array[1]
    dc = dist_array[2]

    de_ind, dm_ind, dc_ind = rank_euc(de, dm, dc)

    os.remove(ranking_output)
    with open("rankings.txt","w") as f:
        f.write("\nRanked Euclidean Distances:\n")
        f.write(str(de[de_ind[1:11]]))
        f.write("\nRanked Manhattan Distances:\n")
        f.write(str(dm[dm_ind[1:11]]))
        f.write("\nRanked Cosine Distances:\n")
        f.write(str(dc[dc_ind[1:11]]))

    save_to_csv(de, "de.csv")
    save_to_csv(dm, "dm.csv")
    save_to_csv(dc, "dc.csv")
