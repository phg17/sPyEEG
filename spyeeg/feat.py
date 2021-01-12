"""
Tools for feature extractions of signal for modelling.

@author: phg17, mak616
"""

import numpy as np
from scipy import signal, fftpack
from sklearn.preprocessing import minmax_scale
import mne


def fast_hilbert(x, axis=0):
    '''
    Fast implementation of Hilbert transform. The trick is to find the next fast
    length of vector for fourier transform (fftpack.helper.next_fast_len(...)).
    Next the matrix of zeros of the next fast length is preallocated and filled
    with the values from the original matrix.

    Inputs:
    - x - input matrix
    - axis - axis along which the hilbert transform should be computed

    Output:
    - x - analytic signal of matrix x (the same shape, but dtype changes to np.complex)
    '''
    # Add dimension if x is a vector
    if len(x.shape) == 1:
        x = x[:, np.newaxis]

    fast_shape = np.array(
        [fftpack.helper.next_fast_len(x.shape[0]), x.shape[1]])
    x_padded = np.zeros(fast_shape)
    x_padded[:x.shape[0], :] = x
    x = signal.hilbert(x_padded, axis=axis)[:x.shape[0], :]
    return x.squeeze()


def filter_signal(x, srate, cutoff, resample=None, rescale=None, **fir_kwargs):
    """Filtering & resampling of a signal through mne create filter function.
    Args:
        x (nd array): Signal as a vector (or array - experimental).
        srate (float): Sampling rate of the signal x in Hz.
        cutoff (float | 2-element list-like): Cutoff frequencies (in Hz).
        resample (float, optional): Sampling rate of the resampled signal in Hz. 
            If None, no resampling. Defaults to None.
        rescale (2 element tuple of floats, optional): Mix-max rescale the signal to the given range. 
            If None, no rescaling. Defaults to None.
        fir_kwargs (optional) - arguments of the mne.filter.create_filter 
        (https://mne.tools/dev/generated/mne.filter.create_filter.html)
    Raises:
        ValueError: Incorrect formatting of input arguments.
        ValueError: Overlap of cutoff frequencies and resmapling freq.
    Returns:
        x [nd array]: Filtered (and optionally resampled) signal

    Example use:
        - Filter audio track to estimate fundamental waveforms for modelling ABR responses.
    """
    if np.isscalar(cutoff):
        l_freq = None
        h_freq = cutoff
    elif len(cutoff) == 2:
        l_freq, h_freq = cutoff
    else:
        raise ValueError(
            "Cutoffs need to be scalar (for low-pass) or 2-element vector (for bandpass).")

    f_nyq = 2*h_freq

    # Design filter
    fir_coefs = mne.filter.create_filter(
        data=x,  # data is only used for sanity checking, not strictly needed
        sfreq=srate,  # sfreq of your data in Hz
        l_freq=l_freq,
        h_freq=h_freq,  # assuming a lowpass of 40 Hz
        method='fir',
        fir_design='firwin',
        **fir_kwargs)

    # Pad & convolve
    x = np.pad(x, (len(fir_coefs) // 2, len(fir_coefs) // 2), mode='edge')
    x = signal.convolve(x, fir_coefs, mode='valid')

    # Resample
    if resample:
        if not f_nyq < resample <= srate:
            raise ValueError(
                "Chose resampling rate more carefully, must be > %.1f Hz" % (f_nyq))
        if srate//resample == srate/resample:
            x = signal.resample_poly(x, 1, srate//resample)
        else:
            dur = (len(x)-1)/srate
            new_n = int(np.ceil(resample * dur))
            x = signal.resample(x, new_n)

    # Scale output between 0 and 1:
    if rescale:
        x = minmax_scale(x, rescale)

    return x


def signal_envelope(x, srate, cutoff=20., resample=None, method='hilbert', comp_factor=1., rescale=None, **fir_kwargs):
    """Extraction of the signal envelope. + filtering and resampling.
    Args:
        x (nd array): Signal as a vector (or array - experimental).
        srate (float): Sampling rate of the signal x in Hz.
        cutoff (float | 2-element list-like, optional): Cutoff frequencies (in Hz). Defaults to 20..
        resample (float, optional): Sampling rate of the resampled signal in Hz. 
            If None, no resampling. Defaults to None.
        method (str, optional): Method for extracting the envelope. 
            Options:
                - hilbert - hilbert transform + abs.
                - rectify - full wave rectification.
            Defaults to 'hilbert'.
        comp_factor (float, optional): Compression factor of the envelope. Defaults to 1..
        rescale (2 element tuple of floats, optional): Mix-max rescale the signal to the given range. 
            If None, no rescaling. Defaults to None.
        fir_kwargs (optional) - arguments of the mne.filter.create_filter 
        (https://mne.tools/dev/generated/mne.filter.create_filter.html)

    Raises:
        NotImplementedError: Envelope extractions methods to be implemented.
        ValueError: Bad format of the argument.

    Returns:
        env [nd array]: Filtered & resampled signal envelope.

    Example use:
        - Extract envelope from speech track for modelling cortical responses.
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
    env = filter_signal(out, srate, cutoff, resample, rescale, **fir_kwargs)

    return env
