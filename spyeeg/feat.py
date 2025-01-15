"""
Tools for feature extractions of signal for modelling.

@author: phg17, mak616
"""

import numpy as np
from scipy import signal, fftpack
from sklearn.preprocessing import minmax_scale
import mne


def signal_envelope(x, srate, cutoff=20., resample=None, method='hilbert', comp_factor=1., rescale=None, verbose=False, **fir_kwargs):
    """
    Extraction of the signal envelope, with filtering and resampling.
    
    Parameters
    ----------
    x : ndarray 
        Signal as a vector.
    srate : float
        Sampling rate of the signal x in Hz.
    cutoff : float | 2-element list-like
        Cutoff frequencies (in Hz). Defaults to 20.
    resample : float
        Sampling rate of the resampled signal in Hz. If None, no resampling. Defaults to None.
    method : str
        Method for extracting the envelope, either 'hilbert' (hilbert transform + abs) or 'rectify' (full wave rectification). Defaults to 'hilbert'.
    comp_factor : float
        Compression factor of the envelope. Defaults to 1 (no compression).
    rescale : tuple of floats
        Mix-max rescale the signal to the given range.
        If None, no rescaling. Defaults to None.
    fir_kwargs : misc
        arguments of the mne.filter.create_filter (https://mne.tools/dev/generated/mne.filter.create_filter.html)

    Raises
    ------
    NotImplementedError: Envelope extractions methods to be implemented.
    ValueError: Bad format of the argument.

    Returns
    -------
    env : ndarray
        Filtered & resampled signal envelope.
    """

    if method.lower() == 'subs':
        raise NotImplementedError
    else:
        if method.lower() == 'hilbert':
            # Get modulus of hilbert transform
            out = abs(fast_hilbert(x))
        elif method.lower() == 'rectify':
            # Rectify x
            out = abs(x)
        else:
            raise ValueError(
                "Method can only be 'hilbert', 'rectify' or 'subs'.")

    # Non linear compression before filtering to avoid NaN
    out = out.astype(np.float)
    out = np.power(out + np.finfo(float).eps, comp_factor)

    # Filtering, resampling
    env = filter_signal(out, srate, cutoff, resample,
                        rescale, verbose=verbose, **fir_kwargs)

    return env


def signal_rectify(signal, mode='half'):
    """Simple rectification method. 
    
    Parameters
    ----------
    signal : ndarray
        Signal to be rectified.
    mode : str
        Rectification mode, either 'hilbert' (hilbert transform + abs) or 'rectify' (full wave rectification). Defaults to 'hilbert'. Defaults to 'half'.
        
    Returns
    -------
    tmp : ndarray
        Rectified signal.
    """
    if mode == 'full':
        return np.abs(signal)
    elif mode == 'half':
        tmp = np.copy(signal)
        tmp[tmp < 0] = 0
        return tmp

def permut_discrete(X):
    """Simple function to permute discrete events i.e. 0 if nothing happens, the values to permute otherwise."""
    X_permut = np.zeros(X.shape)
    values = X[X != 0]
    indices = np.where(X!=0)[0]
    permut_values = np.random.permutation(values)
    for i,v in zip(indices,permut_values):
        X_permut[i] = v
    return X_permut
