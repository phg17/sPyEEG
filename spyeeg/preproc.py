#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:48:37 2020

@author: Pierre Guilleminot
"""

# Libraries
# Standard library
import numpy as np
from scipy import signal
from scipy.linalg import eigh as geigh
from joblib import Parallel, delayed
from sklearn.covariance import oas, ledoit_wolf, fast_mcd, empirical_covariance
from sklearn.preprocessing import scale
import mne


def scale_discrete(X):
    """Small function to z-scores the non-null elements of a discrete time series (i.e. 0 when no events, the values to scale otherwise)."""
    n_feat = X.shape[1]
    Xscale = np.zeros(X.shape)
    for i_feat in range(n_feat):
        Xscale[np.where(X[:,i_feat]>0),i_feat] = scale((X[np.where(X[:,i_feat]>0),i_feat])[0])
    return Xscale

def mix_signal_noise(signal, noise, snr_db):
    """
    Mix a signal and noise according to a specified signal-to-noise ratio (SNR).
    
    Parameters
    ----------
    signal: np.ndarray
        The time series representing the signal.
    noise: np.ndarray
        The time series representing the noise.
    snr_db: float
        The desired signal-to-noise ratio in decibels (dB).
    
    Returns
    -------
    mixed: np.ndarray
        The resulting time series with the signal and noise mixed.
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
