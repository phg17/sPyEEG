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
from mne.filter import filter_data


def simulate_continuous_stimuli(fs, time_array, mode = 'AR', phi = 1.1, noise_std = 0.9):
    """
    Generate an arbitrary time series representing a continuous stimuli. This can be done using either
    convolutions of random sine waves, or an autoregressive(AR) model. 
    Using the autocorrelation methods avoid having stimuli with strong periodicity, which typically creates
    artifacts when fitting the different models.

    Parameters:
        fs (int): The sampling frequency of the signal, in Hz.
        time_array (ndarray): The different timesteps, typically a range from 0 to N-1 for N timepoints.
        mode (str): Methods to generate the arbitrary stimuli. Must be either 'AR' or 'autocorrelation'.
        phi (float): Autoregression coefficient.
        noise_std: The standard deviation of the Gaussian noise used in the AR model.

    Returns:
        ndarray: An arbitrary continuous stimuli.
    """
    if mode == 'convolution':
        #Generate a set of random periodic signals and then convolve them
        
        signal1 = np.random.randint(1,100) * np.sin(2*np.pi*np.random.randint(1,20)*time_array/fs) + np.cos(2*np.pi*np.random.randint(1,80)*time_array/fs) + np.cos(2*np.pi*np.random.randint(1,84)*time_array/fs)
        signal2 = np.random.randint(1,100) * np.sin(2*np.pi*np.random.randint(1,40)*time_array/fs) + np.cos(2*np.pi*np.random.randint(1,60)*time_array/fs) + np.cos(2*np.pi*np.random.randint(1,80)*time_array/fs)
        signal3 = np.random.randint(1,100) * np.sin(2*np.pi*np.random.randint(1,60)*time_array/fs) + np.cos(2*np.pi*np.random.randint(1,40)*time_array/fs) + np.cos(2*np.pi*np.random.randint(1,20)*time_array/fs)
        signal4 = np.random.randint(1,100) * np.sin(2*np.pi*np.random.randint(1,80)*time_array/fs) + np.cos(2*np.pi*np.random.randint(1,20)*time_array/fs) + np.cos(2*np.pi*np.random.randint(1,60)*time_array/fs)
        y1 = convolve(signal1*signal2, signal3*signal4, 'same')
        y2 = convolve(signal1*signal3, signal2*signal4, 'same')
        y = convolve(y1,y2, 'same')
        
    elif mode == 'AR':
        #Generate datapoints step by step using the previous one and add noise.
        
        n_samples = time_array.shape[0]
        phi = 0.9
        noise_std = 0.5
        noise = np.random.normal(0, noise_std, n_samples)
        y = np.zeros(n_samples)
        for t in range(1,n_samples):
            y[t] = phi * y[t-1] + noise[t]
    else:
        raise ValueError(f"Invalid value for 'mode': {mode}. Must be one of AR or convolution.")

    return y

def simulate_channels(n_feat = 2, n_channels = 3, 
                      fs = 100, T = 60, 
                      snr_db = 0, beta_noise = 0,
                      stim_type = 'discrete', n_pulse = 120, share_events = True, 
                      weights_feat = [], weights_channel = [],
                      compression_factor = 1, 
                      impulse_freqs = [0.1,10], decreasing_rates = [0.1,20], delays = [0.06,0.2], filter_impulse = False,
                      share_impulse = False,
                      random_seed = 0, scale_data = True):
    """
    Simulate M/sE/EEG channels as the combination of responses to arbitrary features and noise. 
    This supposedly models a linear time invariant system, considering noise as every process other than 
    the one in response to the stimuli. There is the possibility to change the number of channels, the
    general shape and weights of impulse responses corresponding to different features, noise color and amplitude, 
    as well as to add a non-linear compression factor.
    
    Parameters:
        n_feat (int): The number of features of the stimuli.
        n_channels (int): The number of channels to simulate.
        fs (float): The sampling frequency, in Hz.
        T (float): The duration of the signal to simulate, in s.
        snd_db (float): The signal to noise, in dB. If equal to 0
        beta_noise (float): The parameter used in noise generation. 
                            if 0, equivalent to pure white noise.
                            if 1, equivalent to pure pink noise.
        stim_type (str): whether to use discrete or continuous features. must be either 'discrete' or 'continuous'.
        n_pulse (int): In the case of discrete features, the number of events to consider.
        share_events (bool): whether different features are related to the same set of events.
        weights_feat (list):
        
    """
    np.random.seed(random_seed)
    if len(weights_feat) == 0:
        weights_feat = np.ones(n_feat)/n_feat
    elif len(weights_feat) == n_feat:
        weights_feat = np.asarray(weights_feat)/np.sum(weights_feat)
    else:
        weights_feat = np.ones(n_feat)/n_feat
        print("Weights have incoherent shape relative to number of features, set to equal weights")
    if len(weights_channel) == 0:
        weights_channel = np.ones(n_channels)
    elif len(weights_feat) == n_feat:
        weights_feat = np.asarray(weights_channels)
    else:
        weights_feat = np.ones(n_channels)
        print("Weights have incoherent shape relative to number of features, set to equal weights")
    n_samples = int(T*fs)
    time_array = np.linspace(0,T,n_samples)
    impulse_responses = np.zeros([n_feat, n_channels,n_samples])
    events = np.zeros([n_feat,n_samples])
    nonlinear_events = np.zeros([n_feat,n_samples])
    response = np.zeros([n_channels,n_samples])
    if stim_type == 'discrete':
        if share_events:
            event_pulses = np.random.randint(0,n_samples,n_pulse)
            for i_feat in range(n_feat):
                events[i_feat,event_pulses] = np.random.random(n_pulse)
        else:
            for i_feat in range(n_feat):
                events[i_feat,np.random.randint(0,n_samples,n_pulse)] = np.random.random(n_pulse)
    elif stim_type == 'continuous':
        for i_feat in range(n_feat):
            y = simulate_continuous_stimuli(fs, time_array)
            events[i_feat,:] = MinMaxScaler(feature_range=(-1,1)).fit_transform(y.reshape(-1, 1)).reshape(-1)
    else:
        raise ValueError(f"Invalid value for 'stim_type': {stim_type}. Must be one of discrete or continuous.")

    for i_feat in range(n_feat):
        for i_channel in range(n_channels):
            impulse_responses[i_feat, i_channel,:] = weights_channel[i_channel]*scale(np.roll(np.sin(2*np.pi*np.random.randint(impulse_freqs[0],impulse_freqs[1])*time_array + np.random.rand()*2*np.pi) * np.exp(-time_array*np.random.randint(decreasing_rates[0],decreasing_rates[1])), np.random.randint(int(delays[0]*fs),int(delays[1]*fs)))) / n_samples
            if filter_impulse:
                impulse_responses[i_feat, i_channel,:] = filter_data(impulse_responses[i_feat, i_channel,:],fs,0.01,fs//3, verbose = False)
            if share_impulse:
                impulse_responses[i_feat, i_channel,:] = weights_channel[i_channel]*impulse_responses[i_feat, 0,:]
                        
    X = events.T
    if scale_data:
        if stim_type == 'continuous':
            X = scale(X,axis = 0)
        else:
            X = scale_discrete(X)
    for i_channel in range(n_channels):
        for i_feat in range(n_feat):
            nonlinear_events[i_feat,:] = np.power(np.abs(events[i_feat,:]), compression_factor) * np.sign(events[i_feat,:])
            response[i_channel] += weights_feat[i_feat]*convolve(nonlinear_events[i_feat,:], impulse_responses[i_feat, i_channel,:])[:n_samples]
        noise = cn.powerlaw_psd_gaussian(beta_noise, n_samples)
        response[i_channel] = mix_signal_noise(response[i_channel], noise, snr_db)

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


import numpy as np

def mix_signal_noise(signal, noise, snr_db):
    """
    Mix a signal and noise according to a specified signal-to-noise ratio (SNR).
    
    Parameters:
    - signal (np.ndarray): The time series representing the signal.
    - noise (np.ndarray): The time series representing the noise.
    - snr_db (float): The desired signal-to-noise ratio in decibels (dB).
    
    Returns:
    - mixed (np.ndarray): The resulting time series with the signal and noise mixed.
    """
    # Ensure signal and noise have the same length
    if len(signal) != len(noise):
        raise ValueError("Signal and noise must have the same length.")
    
    # Compute the power of the signal and noise
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    
    # Compute the scaling factor for the noise based on the desired SNR
    snr_linear = 10 ** (snr_db / 10)  # Convert SNR from dB to linear scale
    scaling_factor = np.sqrt(signal_power / (noise_power * snr_linear))
    
    # Scale the noise and mix it with the signal
    scaled_noise = noise * scaling_factor
    mixed = signal + scaled_noise
    
    return mixed





        
        


