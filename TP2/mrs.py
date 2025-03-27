"""
Created on Tue Apr  6 13:03:06 2021

@author: rpp
"""

import librosa #https://librosa.org/    #sudo apt-get install -y ffmpeg (open mp3 files)
import librosa.display
import librosa.beat
import sounddevice as sd  #https://anaconda.org/conda-forge/python-sounddevice
import warnings
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.close('all')
    
    #--- Load file
    fName = "../Queries/MT0000414517.mp3"    
    sr = 22050
    mono = True
    warnings.filterwarnings("ignore")
    y, fs = librosa.load(fName, sr=sr, mono = mono)
    print(y.shape)
    print(fs)
    
    #--- Play Sound
    #sd.play(y, sr, blocking=False)
    
    #--- Plot sound waveform
    plt.figure()
    librosa.display.waveshow(y)
    
    #--- Plot spectrogram
    Y = np.abs(librosa.stft(y))
    Ydb = librosa.amplitude_to_db(Y, ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(Ydb, y_axis='linear', x_axis='time', ax=ax)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
        
    #--- Extract features    
    sc = librosa.feature.spectral_centroid(y = y)  #default parameters: sr = 22050 Hz, mono, window length = frame length = 92.88 ms e hop length = 23.22 ms 
    sc = sc[0, :]
    print(sc.shape)
    times = librosa.times_like(sc)
    plt.figure(), plt.plot(times, sc)
    plt.xlabel('Time (s)')
    plt.title('Spectral Centroid')
    