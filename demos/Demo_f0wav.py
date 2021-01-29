"""
Demo showing how to use the f0 waveform from spyeeg.feat module.

@author: mak616
"""
import numpy as np
import scipy.io as sio
import spyeeg

# Load audio sample
fs_audio, audio = sio.wavfile.read('./Data/audio_sample.wav')

# Load fundamental waveform
f0wav = sio.loadmat('./Data/f0wav_sample.mat')['F0'].squeeze()
fs_f0wav = 8820  # Sampling rate of fund. waveform

# Estimate fundamental wavefrom from audio
f0wav_hat = spyeeg.feat.signal_f0wav(audio, srate=fs_audio, resample=fs_f0wav)

# Print correlation coef. of the actual FW and estimate
print('Correlation coef. (actual vs estimated): {:.3f}'.format(
    np.corrcoef(f0wav, f0wav_hat)[0, 1]))
