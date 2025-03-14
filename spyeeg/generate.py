"""
Created on Thu Nov  14 18:32:12 2024

@author: Pierre Guilleminot
"""

import numpy as np
from scipy import signal, fftpack
import scipy.signal as signal
from sklearn.preprocessing import scale
from scipy.signal import convolve
from sklearn.preprocessing import MinMaxScaler, scale
import colorednoise as cn
from .preproc import scale_discrete, mix_signal_noise
from mne.filter import filter_data

def simulate_PAC(k_pac,time_array,lf,hf):
    """
    Generate a simple pac time series between two sine waves according to a coupling coefficient k_pac.

    Parameters
    ----------
    k_pac : float or ndarray
        Coupling coefficient, can either be stable along time (float), or change over time, in which case it is an ndarray of shape (Ntimes)
    time_array : ndarray
        Array of times, of shape (Ntimes)
    lf : float
        low-frequency of the PAC, from which we extract the phase
    hf : float
        high-frequency to be modulated
    """
    s1 = np.cos(2*np.pi*lf*time_array)
    s1 -= np.min(s1)
    s2 = np.cos(2*np.pi*hf*time_array)
    phase = np.angle(signal.hilbert(s1))
    amplitude = k_pac * np.cos(phase)
    pac = amplitude * s2
    return pac

def simulate_continuous_stimuli(fs, time_array, mode = 'AR', phi = 1.1, noise_std = 0.9):
    """
    Generate an arbitrary time series representing a continuous stimuli. This can be done using either
    convolutions of random sine waves, or an autoregressive(AR) model. 
    Using the autocorrelation methods avoid having stimuli with strong periodicity, which typically creates
    artifacts when fitting the different models.

    Parameters
    ----------
    fs : int 
        The sampling frequency of the signal, in Hz.
    time_array : ndarray
        The different timesteps, typically a range from 0 to N-1 for N timepoints.
    mode : str
        Methods to generate the arbitrary stimuli. Must be either 'AR' or 'autocorrelation'.
    phi : float
        Autoregression coefficient.
    noise_std : float
        The standard deviation of the Gaussian noise used in the AR model.

    Returns
    -------
    y : ndarray 
        An arbitrary continuous stimuli.
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
                      impulse_freqs = [0.1,10], decreasing_rates = [0.1,20], delays = [0.06,0.2], 
                      filter_impulse = False, filter_val = [0.01,20],
                      share_impulse = False,
                      random_seed = 0, scale_data = True,
                      manual_events = None,
                      target_signal = 'raw', PAC_lf = 1, PAC_hf = 20):
    """
    Simulate M/sE/EEG channels as the combination of responses to arbitrary features and noise. 
    This supposedly models a linear time invariant system, considering noise as every process other than 
    the one in response to the stimuli. There is the possibility to change the number of channels, the
    general shape and weights of impulse responses corresponding to different features, noise color and amplitude, 
    as well as to add a non-linear compression factor.
    The impulse responses are all in the forms of a sine wave with an exponential decay, shifted in time.
    This functions returns the timepoints, features, channels and impulse responses.
    
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
        share_events (bool): Whether different features are related to the same set of events.
        weights_feat (list): List of weights attributed to different features.
        weights_channel (list): List of weights, i.e. signal strength, on different channels. These are the same for every features.
        compression_factor (float): Exponent factor linking feature and impulse response. 
                                    This is used to test the effect of the linearity assumption violation.
        impulse_freqs (list): Minimum and maximum frequency possible for the frequency of the impulse response.
        decreasing_rates (list): Minimum and maximum exponential decay of the impulse response.
        delays (list): Minimum and maximum delays to shift the impulse response by. 
                       Postive values mean the brain response happen AFTER the stimuli, which is the causal direction.
                       Negative values could be interpreted as a prediction from the brain.
        filter_impulse (bool): Whether to filter the neural data.
        filter_val (list): Low pass and high-pass values of the filter to apply.
        share_impulse (bool): Whether different channels respond to the stimuli with the same impulse.
                              Typically True for EEG as channels share common signals. 
                              Depending on implantation, this could not be the case sEEG for example.
        random seed (int): Random seed.
        scale_data (bool): Whether to z-scores the data. 
        
    Returns:
        time_array (np.ndarray): Array of timepoints.
        X (np.ndarray): Feature/Stimuli matrix.
        Y (np.ndarray): Neural data matrix.
        events (np.ndarray): Features/Stimuli. 
        impulse_responses (np.ndarray): Impulse responses.
        
    """
    np.random.seed(random_seed)

    # Set up weights if necessary
    if len(weights_feat) == 0:
        weights_feat = np.ones(n_feat)/n_feat
    elif len(weights_feat) == n_feat:
        weights_feat = np.asarray(weights_feat)/np.sum(weights_feat)
    else:
        weights_feat = np.ones(n_feat)/n_feat
        print("Weights have incoherent shape relative to number of features, set to equal weights")
    if len(weights_channel) == 0:
        weights_channel = np.ones(n_channels)
    elif len(weights_channel) == n_channels:
        weights_channel = np.asarray(weights_channel)
    else:
        weights_channel = np.ones(n_channels)
        print("Weights have incoherent shape relative to number of features, set to equal weights")

    #Set up arrays
    n_samples = int(T*fs)
    time_array = np.linspace(0,T,n_samples)
    impulse_responses = np.zeros([n_feat, n_channels,n_samples])
    events = np.zeros([n_feat,n_samples])
    nonlinear_events = np.zeros([n_feat,n_samples])
    response = np.zeros([n_channels,n_samples])

    #Generate features i.e. stimuli
    if manual_events == None:
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
    else:
        events = manual_events

    #Generate Impulse Response
    for i_feat in range(n_feat):
        for i_channel in range(n_channels):
            impulse_responses[i_feat, i_channel,:] = weights_channel[i_channel]*scale(np.roll(np.sin(2*np.pi*np.random.randint(impulse_freqs[0],impulse_freqs[1])*time_array + np.random.rand()*2*np.pi) * np.exp(-time_array*np.random.randint(decreasing_rates[0],decreasing_rates[1])), np.random.randint(int(delays[0]*fs),int(delays[1]*fs)))) / n_samples
            if filter_impulse:
                impulse_responses[i_feat, i_channel,:] = filter_data(impulse_responses[i_feat, i_channel,:],fs,filter_val[0],filter_val[1], verbose = False)
            if share_impulse:
                impulse_responses[i_feat, i_channel,:] = weights_channel[i_channel]*impulse_responses[i_feat, 0,:]

    #Scale stimuli
    X = events.T
    if scale_data:
        X = scale(X,axis = 0)

    #Generate channels as the convolution between features and impulse responses
    for i_channel in range(n_channels):
        for i_feat in range(n_feat):
            nonlinear_events[i_feat,:] = np.power(np.abs(events[i_feat,:]), compression_factor) * np.sign(events[i_feat,:])
            if target_signal == 'raw':
                response[i_channel] += weights_feat[i_feat]*convolve(nonlinear_events[i_feat,:], impulse_responses[i_feat, i_channel,:])[:n_samples]
            elif target_signal == 'PAC':
                response_feat = weights_feat[i_feat]*convolve(nonlinear_events[i_feat,:], impulse_responses[i_feat, i_channel,:])[:n_samples]
                pac_feat = simulate_PAC(response_feat, time_array, PAC_lf, PAC_hf)
                response[i_channel] += pac_feat
            else:
                raise "Not a valid target"
        noise = cn.powerlaw_psd_gaussian(beta_noise, n_samples)
        response[i_channel] = mix_signal_noise(response[i_channel], noise, snr_db)

    #scale channels
    Y = response.T
    if scale_data:
        Y = scale(Y, axis = 0)

    return time_array, X, Y, events, impulse_responses

def simulate_multisensory_channels(n_feat = 1, n_channels = 1, 
                      fs = 100, T = 60, 
                      snr_db = 0, beta_noise = 0,
                      stim_type = 'continuous', n_pulse = 120, 
                      compression_factor = 1, 
                      impulse_freqs = [0.1,10], decreasing_rates = [0.1,20], delays = [0.06,0.2],
                      random_seed = 0, scale_data = True, supra_amp = 1):
    """
    Simulate multisensory M/sE/EEG channels as the combination of responses to arbitrary features and noise, for two modalities. 
    As previously, this models a linear time invariant system with the added constraint that for the same feature, there are different
    responses possible corresponding to the unisensory responses of the two possible modalities, or the multisensory response.
    Here, we assume a supra-linear effect in the form of an additional impulse response when both modalities are present.
    
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
        share_events (bool): Whether different features are related to the same set of events.
        weights_feat (list): List of weights attributed to different features.
        weights_channel (list): List of weights, i.e. signal strength, on different channels. These are the same for every features.
        compression_factor (float): Exponent factor linking feature and impulse response. 
                                    This is used to test the effect of the linearity assumption violation.
        impulse_freqs (list): Minimum and maximum frequency possible for the frequency of the impulse response.
        decreasing_rates (list): Minimum and maximum exponential decay of the impulse response.
        delays (list): Minimum and maximum delays to shift the impulse response by. 
                       Postive values mean the brain response happen AFTER the stimuli, which is the causal direction.
                       Negative values could be interpreted as a prediction from the brain.
        filter_impulse (bool): Whether to filter the neural data.
        filter_val (list): Low pass and high-pass values of the filter to apply.
        share_impulse (bool): Whether different channels respond to the stimuli with the same impulse.
                              Typically True for EEG as channels share common signals. 
                              Depending on implantation, this could not be the case sEEG for example.
        random seed (int): Random seed.
        scale_data (bool): Whether to z-scores the data. 
        
    """
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
                noise = cn.powerlaw_psd_gaussian(beta_noise, n_samples)
                nonlinear_events[i_feat,:] = np.power(np.abs(events[i_feat,:]), compression_factor) * np.sign(events[i_feat,:])
                response[i_modality,i_channel] += convolve(nonlinear_events[i_feat,:], impulse_responses[i_modality,i_feat, i_channel,:])[:n_samples]
                response[i_modality,i_channel] = mix_signal_noise(response[i_modality,i_channel], noise, snr_db)
        
    Y1, Y2, Y12 = response[0].T, response[1].T, response[2].T
    if scale_data:
        Y1, Y2, Y12 = scale(Y1, axis = 0), scale(Y2, axis = 0), scale(Y12, axis = 0)
        
    return time_array, X, Y1,Y2,Y12, events, impulse_responses







        
        


