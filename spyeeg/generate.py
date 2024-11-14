"""
Created on Thu Nov  14 18:32:12 2024

@author: phg17
"""

import numpy as np
from scipy import signal, fftpack
import scipy.signal as signal
from sklearn.preprocessing import scale
from scipy.signal import convolve
from sklearn.preprocessing import MinMaxScaler, scale
import colorednoise as cn
from .preproc import scale_discrete


def simulate_continuous_stimuli(fs, time_array, mode = 'AR', phi = 1.1, noise_std = 0.9):
    if mode == 'convolution':
        signal1 = np.random.randint(1,100) * np.sin(2*np.pi*np.random.randint(1,20)*time_array/fs) + np.cos(2*np.pi*np.random.randint(1,80)*time_array/fs) + np.cos(2*np.pi*np.random.randint(1,84)*time_array/fs)
        signal2 = np.random.randint(1,100) * np.sin(2*np.pi*np.random.randint(1,40)*time_array/fs) + np.cos(2*np.pi*np.random.randint(1,60)*time_array/fs) + np.cos(2*np.pi*np.random.randint(1,80)*time_array/fs)
        signal3 = np.random.randint(1,100) * np.sin(2*np.pi*np.random.randint(1,60)*time_array/fs) + np.cos(2*np.pi*np.random.randint(1,40)*time_array/fs) + np.cos(2*np.pi*np.random.randint(1,20)*time_array/fs)
        signal4 = np.random.randint(1,100) * np.sin(2*np.pi*np.random.randint(1,80)*time_array/fs) + np.cos(2*np.pi*np.random.randint(1,20)*time_array/fs) + np.cos(2*np.pi*np.random.randint(1,60)*time_array/fs)
        y1 = convolve(signal1*signal2, signal3*signal4, 'same')
        y2 = convolve(signal1*signal3, signal2*signal4, 'same')
        y = convolve(y1,y2, 'same')
    elif mode == 'AR':
        n_samples = time_array.shape[0]
        phi = 0.9
        noise_std = 0.5
        noise = np.random.normal(0, noise_std, n_samples)
        y = np.zeros(n_samples)
        for t in range(1,n_samples):
            y[t] = phi * y[t-1] + noise[t]

    return y

def simulate_channels(n_feat = 2, n_channels = 3, 
                      fs = 100, T = 60, 
                      noise_level = 0, beta_noise = 0,
                      stim_type = 'discrete', n_pulse = 120, 
                      compression_factor = 1, 
                      impulse_freqs = [0.1,10], decreasing_rates = [0.1,20], delays = [0.06,0.2],
                      random_seed = 0, scale_data = True):
    np.random.seed(random_seed)
    n_samples = int(T*fs)
    time_array = np.linspace(0,T,n_samples)
    impulse_responses = np.zeros([n_feat, n_channels,n_samples])
    events = np.zeros([n_feat,n_samples])
    nonlinear_events = np.zeros([n_feat,n_samples])
    response = np.zeros([n_channels,n_samples])
    if stim_type == 'discrete':
        for i_feat in range(n_feat):
            events[i_feat,np.random.randint(0,n_samples,n_pulse)] = np.random.random(n_pulse)
    elif stim_type == 'continuous':
        for i_feat in range(n_feat):
            signal1 = np.random.randint(1,100) * np.sin(2*np.pi*np.random.randint(1,100)*time_array/fs)*np.cos(2*np.pi*np.random.randint(1,100)*time_array/fs)**2
            signal2 = np.random.randint(1,100) * np.sin(2*np.pi*np.random.randint(1,100)*time_array/fs)*np.cos(2*np.pi*np.random.randint(1,100)*time_array/fs)**2
            y = convolve(signal1, signal2, 'same')
            y = simulate_continuous_stimuli(fs, time_array)
            events[i_feat,:] = MinMaxScaler(feature_range=(-1,1)).fit_transform(y.reshape(-1, 1)).reshape(-1)

    for i_feat in range(n_feat):
        for i_channel in range(n_channels):
            impulse_responses[i_feat, i_channel,:] = np.roll(np.sin(2*np.pi*np.random.randint(impulse_freqs[0],impulse_freqs[1])*time_array + np.random.rand()*2*np.pi) * np.exp(-time_array*np.random.randint(decreasing_rates[0],decreasing_rates[1])), np.random.randint(int(delays[0]*fs),int(delays[1]*fs)))
                        
    X = events.T
    if scale_data:
        if stim_type == 'continuous':
            X = scale(X,axis = 0)
        else:
            X = scale_discrete(X)

    for i_feat in range(n_feat):
        for i_channel in range(n_channels):
            noise = cn.powerlaw_psd_gaussian(beta_noise, n_samples) * noise_level
            nonlinear_events[i_feat,:] = np.power(np.abs(events[i_feat,:]), compression_factor) * np.sign(events[i_feat,:])
            response[i_channel] += convolve(nonlinear_events[i_feat,:], impulse_responses[i_feat, i_channel,:])[:n_samples] + noise
    
    Y = response.T
    if scale_data:
        Y = scale(Y, axis = 0)

    return time_array, X, Y, events, impulse_responses

def simulate_multisensory_channels(n_feat = 1, n_channels = 1, 
                      fs = 100, T = 60, 
                      noise_level = 0, beta_noise = 0,
                      stim_type = 'continuous', n_pulse = 120, 
                      compression_factor = 1, 
                      impulse_freqs = [0.1,10], decreasing_rates = [0.1,20], delays = [0.06,0.2],
                      random_seed = 0, scale_data = True, supra_amp = 1):
    np.random.seed(random_seed)
    n_modality = 3
    n_samples = int(T*fs)
    time_array = np.linspace(0,T,n_samples)
    impulse_responses = np.zeros([n_modality,n_feat, n_channels,n_samples])
    events = np.zeros([n_feat,n_samples])
    nonlinear_events = np.zeros([n_feat,n_samples])
    response = np.zeros([n_modality,n_channels,n_samples])
    if stim_type == 'discrete':
        for i_feat in range(n_feat):
            events[i_feat,np.random.randint(0,n_samples,n_pulse)] = np.random.random(n_pulse)
    elif stim_type == 'continuous':
        for i_feat in range(n_feat):
            y = simulate_continuous_stimuli(fs, time_array)
            events[i_feat,:] = MinMaxScaler(feature_range=(-1,1)).fit_transform(y.reshape(-1, 1)).reshape(-1)
    
    for i_feat in range(n_feat):
        for i_channel in range(n_channels):
            for i_modality in range(2):
                impulse_responses[i_modality, i_feat, i_channel,:] = np.roll(np.sin(2*np.pi*np.random.randint(impulse_freqs[0],impulse_freqs[1])*time_array + np.random.rand()*2*np.pi) * np.exp(-time_array*np.random.randint(decreasing_rates[0],decreasing_rates[1])), np.random.randint(int(delays[0]*fs),int(delays[1]*fs)))
            supraadditive_impulse = supra_amp * np.roll(np.sin(2*np.pi*np.random.randint(impulse_freqs[0],impulse_freqs[1])*time_array + np.random.rand()*2*np.pi) * np.exp(-time_array*np.random.randint(decreasing_rates[0],decreasing_rates[1])), np.random.randint(int(delays[0]*fs),int(delays[1]*fs)))
            impulse_responses[2, i_feat, i_channel,:] = np.sum(impulse_responses[:2, i_feat, i_channel,:],axis=0) + supraadditive_impulse
    X = events.T

    if scale_data:
        if stim_type == 'continuous':
            X = scale(X,axis = 0)
        else:
            X = scale_discrete(X)

    for i_feat in range(n_feat):
        for i_channel in range(n_channels):
            for i_modality in range(3):
                noise = cn.powerlaw_psd_gaussian(beta_noise, n_samples) * noise_level
                nonlinear_events[i_feat,:] = np.power(np.abs(events[i_feat,:]), compression_factor) * np.sign(events[i_feat,:])
                response[i_modality,i_channel] += convolve(nonlinear_events[i_feat,:], impulse_responses[i_modality,i_feat, i_channel,:])[:n_samples] + noise
        
    Y1, Y2, Y12 = response[0].T, response[1].T, response[2].T
    if scale_data:
        Y1, Y2, Y12 = scale(Y1, axis = 0), scale(Y2, axis = 0), scale(Y12, axis = 0)
        
    return time_array, X, Y1,Y2,Y12, events, impulse_responses




        
        


