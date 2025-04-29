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

musicsfolder = "Music/"
ranking_output = "rankings.txt"
notNorm = "validação de resultados_TP2/notNormFM_All.csv"  

def extract_features(folder):
    feature = [0] * 190
    sr = 22050

    y, sr = librosa.load(folder, sr= sr)
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

    fzero = librosa.yin(y=y ,fmin = librosa.note_to_hz('C2'), fmax = librosa.note_to_hz('C7'))

    for i in range(fzero.shape[0]):
        if fzero[i] > sr/2:
            fzero[i] = 0

    #fzero[fzero > fs/2] = 0
    
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


def save_before_normalizing(features_list, output_file="notNormTest.csv"):
    np.savetxt(output_file, features_list, "%.6f", delimiter=",")
    print(f"Features saved to {output_file}")

def save_to_csv(features_list, output_file):
    np.savetxt(output_file, features_list, "%.6f", delimiter=", ")
    print(f"Features saved to {output_file}")


def normalize(value ,col_min, col_max):
    new_value = (value - col_min) / (col_max - col_min)
    return new_value


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

if __name__== "__main__":
    music_test = 900
    musics = os.listdir(musicsfolder)
    features = None
    dist_calc = 1
    dist_array = np.zeros((3, 900))

    fm_Q = pd.read_csv('validação de resultados_TP2/FM_Q.csv')

    fm_all = pd.read_csv('validação de resultados_TP2/FM_All.csv')

    array_erros = []
    
    for music in musics:
         
        array_e = []
        if music_test == 0:
            break

        print(f"Processing: {music}, {music_test} musics left...")
        path = os.path.join(musicsfolder, music)

        
        feature = extract_features(path)
        print(f"Número de features extraídas: {len(feature)}")
        
        if type(features) == NoneType:
            features = np.array(feature)
        else:
            features = np.vstack((features, feature))


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

    save_before_normalizing(features)

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
    save_to_csv(features, "audio_features.csv")

    save_to_csv(array_erros, "sc.csv")

    de = dist_array[0]
    dm = dist_array[1]
    dc = dist_array[2]

    de_ind, dm_ind, dc_ind = rank_euc(de, dm, dc)

    try:
        os.remove(ranking_output)
        with open("rankings.txt","w") as f:
            f.write("\nRanked Euclidean Distances:\n")
            f.write(str(de[de_ind[1:11]]))
            f.write("\nRanked Manhattan Distances:\n")
            f.write(str(dm[dm_ind[1:11]]))
            f.write("\nRanked Cosine Distances:\n")
            f.write(str(dc[dc_ind[1:11]]))
        
    except:
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

    #meta_query=np.genfromtxt(StringIO("query_metadata.csv"), delimiter=",")
