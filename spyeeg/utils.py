#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 12:21:04 2020

@author: phg17
"""

# Libraries
import psutil
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import signal
from sklearn.preprocessing import minmax_scale
import pandas as pd
import matplotlib.pyplot as plt

def audio_to_float(audio):
    """Simple remapping of soundfiles in PCM int to float
    Args:
        audio (1D array): sound (PCM int)
    Returns:
        audio (1D array): sound (PCM float)
    """
    iinfo = np.iinfo(audio.dtype)
    max_val = max(abs(iinfo.min), abs(iinfo.max))
    audio = audio/max_val
    return audio

def RMS(x):
    return np.sqrt(np.mean(np.power(x, 2)))


def AddNoisePostNorm(Target, Noise, SNR, spacing=0):
    l_Target = len(Target)
    rmsS = 1
    rmsN = rmsS*(10**(-SNR/20.))
    insert = 0
    Noise = Noise[insert:insert + 2 * spacing + l_Target]
    Noise = Noise * rmsN
    Target_Noise = Noise
    Target_Noise[spacing:spacing + l_Target] += Target
    Target_Noise = Target_Noise / RMS(Target_Noise)

    return Target_Noise


def print_title(msg, line='=', frame=True):
    """Printing function, allowing to print a titled message (underlined or framded)

    Parameters
    ----------
    msg : str
        String of characters
    line : str
        Which character to use to underline (default "=")
    frame : bool
        Whether to frame or only underline title
    """
    print((line*len(msg)+"\n" if frame else "") + msg + '\n'+line*len(msg)+'\n')


def lag_matrix(data, lag_samples=(-1, 0, 1), filling=np.nan, drop_missing=False):
    """Helper function to create a matrix of lagged time series.

    The lag can be arbitrarily spaced. Check other functions to create series of lags
    whether they are contiguous or sparsely spanning a time window :func:`lag_span` and
    :func:`lag_sparse`.

    Parameters
    ----------
    data : ndarray (nsamples x nfeats)
        Multivariate data
    lag_samples : list
        Shift in _samples_ to be applied to data. Negative shifts are lagged in the past,
        positive shits in the future, and a shift of 0 represents the data array as it is
        in the input `data`.
    filling : float
        What value to use to fill entries which are not defined (Default: NaN).
    drop_missing : bool
        Whether to drop rows where filling occured.

    Returns
    -------
    lagged : ndarray (nsamples_new x nfeats*len(lag_samples))
        Matrix of lagged time series.

    Raises
    ------
    ValueError
        If ``filling`` is set by user and ``drop_missing`` is ``True`` (it should be one or
        the other, the error is raised to avoid this confusion by users).

    Example
    -------
    >>> data = np.asarray([[1,2,3,4,5,6],[7,8,9,10,11,12]]).T
    >>> out = lag_matrix(data, (0,1))
    >>> out
    array([[ 1.,  7.,  2.,  8.],
            [ 2.,  8.,  3.,  9.],
            [ 3.,  9.,  4., 10.],
            [ 4., 10.,  5., 11.],
            [ 5., 11.,  6., 12.],
            [ 6., 12., nan, nan]])

    """
    if not np.isnan(filling) and drop_missing:
        raise ValueError(
            "Dropping missing values or filling them are two mutually exclusive arguments!")

    dframe = pd.DataFrame(data)

    cols = []
    for lag in lag_samples:
        # cols.append(dframe.shift(-lag))
        cols.append(dframe.shift(lag))

    dframe = pd.concat(cols, axis=1)
    dframe.fillna(filling, inplace=True)
    if drop_missing:
        dframe.dropna(inplace=True)

    return dframe.values
    # return dframe.loc[:, ::-1].get_values()


def lag_span(tmin, tmax, srate=125):
    """Create an array of lags spanning the time window [tmin, tmax].

    Parameters
    ----------
    tmin : float
        In seconds
    tmax : float
    srate : float
        Sampling rate

    Returns
    -------
    lags : 1d array
        Array of lags in _samples_

    """
    sample_min, sample_max = int(
        np.ceil(tmin * srate)), int(np.ceil(tmax * srate))
    return np.arange(sample_min, sample_max)


def lag_sparse(times, srate=125):
    """Create an array of lags for the requested time point in `times`.

    Parameters
    ----------
    times : list
        List of time point in seconds
    srate : float
        Sampling rate

    Returns
    -------
    lags : 1d array
        Array of lags in _samples_

    """
    return np.asarray([int(np.ceil(t * srate)) for t in times])


def fir_order(tbw, srate, atten=60., ripples=None):
    """Estimate FIR Type II filter order (order will be odd).

    If ripple is given will use rule:

    .. math ::

        N = \\frac{2}{3} \log_{10}\\frac{1}{10\delta_ripp\delta_att} \\frac{Fs}{TBW}

    Else:

    .. math ::

        N = \\frac{Atten*Fs}{22*TBW} - 1

    Parameters
    ----------
    tbw : float
        Transition bandwidth in Hertz
    srate : float
        Sampling rate (Fs) in Hertz
    atten : float (default 60.0)
        Attenuation in StopBand in dB
    ripples : float (default None, optional)
        Maximum ripples height (in relative to peak)

    Returns
    -------
    order : int
        Filter order (i.e. 1+numtaps)

    Notes
    -----
    Rule of thumbs from here_.

    .. _here : https://dsp.stackexchange.com/a/31077/28372
    """
    if ripples:
        atten = 10**(-abs(atten)/10.)
        order = 2./3.*np.log10(1./10/ripples/atten) * srate / tbw
    else:
        order = (atten * srate) / (22. * tbw)

    order = int(order)
    # be sure to return odd order
    return order + (order % 2-1)


def _is_1d(arr):
    "Short utility function to check if an array is vector-like"
    return np.product(arr.shape) == max(arr.shape)


def is_pos_def(A):
    """Check if matrix is positive definite
    Ref: https://stackoverflow.com/a/44287862/5303618
    """
    if np.array_equal(A, A.conj().T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def rolling_func(func, data, winsize=2, overlap=1, padding=True):
    """Apply a function on a rolling window on the data

    Args:
        func ([type]): [description]
        data ([type]): [description]
        winsize (int, optional): [description]. Defaults to 2.
        overlap (int, optional): [description]. Defaults to 1.
        padding (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    # TODO: check when Parallel()(delayed(func)(x) for x in rolled_array)
    # becomes advantageous, because for now it seemed actually slower...
    # (tested only with np.cov on 15min of 64 EEG at 125Hx, list comprehension still faster)
    return [func(x) for x in chunk_data(data, win_as_samples=True, window_size=winsize, overlap_size=overlap, padding=True).swapaxes(1, 2)]


def moving_average(data, winsize=2):
    """TODO

    Args:
        data ([type]): [description]
        winsize (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    # TODO: pad before calling chunk_data?
    return chunk_data(data, window_size=winsize, overlap_size=(winsize-1)).mean(1)


def shift_array(arr, win=2, overlap=0, padding=False, axis=0):
    """Returns segments of an array (overlapping moving windows)
    using the `as_strided` function from NumPy.

    Parameters
    ----------
    arr : numpy.ndarray
    win : int
        Number of samples in one window
    overlap : int
        Number of samples overlapping (0 means no overlap)
    pad : function
        padding function to be applied to data (if False
        will throw away data)
    axis : int
        Axis on which to apply the rolling window

    Returns
    -------
    shiftarr : ndarray
        Shifted copies of array segments

    See Also
    --------
    :func:`pyeeg.utils.chunk_data`

    Notes
    -----
    Using the `as_strided` function trick from Numpy means the returned
    array share the same memory buffer as the original array, so use
    with caution!
    Maybe `.copy()` the result if needed.
    This is the way for 2d array with overlap (i.e. step size != 1, which was the easy way):

    .. code-block:: python

        as_strided(a, (num_windows, win_size, n_features), (size_onelement * hop_size * n_feats, original_strides))
    """
    n_samples = len(arr)
    if not (1 < win < n_samples):
        raise ValueError(
            "window size must be greater than 1 and smaller than len(input)")
    if overlap < 0 or overlap > win:
        raise ValueError(
            "Overlap size must be a positive integer smaller than window size")

    if padding:
        raise NotImplementedError(
            "As a workaround, please pad array beforehand...")

    if not _is_1d(arr):
        if axis != 0:
            arr = np.swapaxes(arr, 0, axis)
        return chunk_data(arr, win, overlap, padding)

    return as_strided(arr, (win, n_samples - win + 1), (arr.itemsize, arr.itemsize))


def chunk_data(data, window_size, overlap_size=0, padding=False, win_as_samples=True):
    """Nd array version of :func:`shift_array`

    Notes
    -----
    Please note that we expect first dim as our axis on which to apply
    the rolling window.
    Calling :func:`mean(axis=0)` works if ``win_as_samples`` is set to ``False``,
    otherwise use :func:`mean(axis=1)`.

    """
    assert data.ndim <= 2, "Data must be 2D at most!"
    if data.ndim == 1:
        data = data.reshape((-1, 1))

    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] -
                   window_size) // (window_size - overlap_size) + 1
    overhang = data.shape[0] - (num_windows *
                                window_size - (num_windows-1) * overlap_size)

    # if there's overhang, need an extra window and a zero pad on the data
    # (numpy 1.7 has a nice pad function I'm not using here)
    # Or should I just NOT add this extra window?
    # The padding beforhand make it clear that we can handle edge values...
    if overhang != 0 and padding:
        num_windows += 1
        #newdata = np.zeros((num_windows * window_size - (num_windows-1) * overlap_size, data.shape[1]))
        #newdata[:data.shape[0]] = data
        #data = newdata
        data = np.pad(data, [(0, overhang+1), (0, 0)], mode='edge')

    size_item = data.dtype.itemsize

    if win_as_samples:
        ret = as_strided(data,
                         shape=(num_windows, window_size, data.shape[1]),
                         strides=(size_item * (window_size - overlap_size) * data.shape[1],) + data.strides)
    else:
        ret = as_strided(data,
                         shape=(window_size, num_windows, data.shape[1]),
                         strides=(data.strides[0], size_item * (window_size - overlap_size) * data.shape[1], data.strides[1]))

    return ret


def find_knee_point(x, y, tol=0.95, plot=False):
    """Function to find elbow or knee point (minimum local curvature) in a curve.
    To do so we look at the angles between adjacent segments formed by triplet of
    points.

    Parameters
    ----------
    x : 1darray
        x- coordinate of the curve
    y : 1darray
        y- coordinate of the curve
    plot : bool (default: False)
        Whether to plot the result

    Returns
    -------
    float
        The x-value of the point of maximum curvature

    Notes
    -----
    The function only works well on smooth curves.
    """
    y = np.asarray(y).copy()
    y -= y.min()
    y /= y.max()
    coords = np.asarray([x, y]).T
    def local_angle(v1, v2): return np.arctan2(
        v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    angles = []
    for k, coord in enumerate(coords[1:-1]):
        v1 = coords[k] - coord
        v2 = coords[k+2] - coord
        angles.append(local_angle(v1, v2))

    if plot:
        plt.plot(x[1:-1], minmax_scale(np.asarray(angles)/np.pi), marker='o')
        plt.hlines(tol, xmin=x[0], xmax=x[-1])
        plt.vlines(x[np.argmin(minmax_scale(np.asarray(angles)/np.pi)
                               <= tol) + 1], ymin=0, ymax=1., linestyles='--')

    return x[np.argmin(minmax_scale(np.asarray(angles)/np.pi) <= tol) + 1]


def mem_check(units='Gb'):
    "Get available RAM"
    stats = psutil.virtual_memory()
    units = units.lower()
    if units == 'gb':
        factor = 1./1024**3
    elif units == 'mb':
        factor = 1./1024**2
    elif units == 'kb':
        factor = 1./1024
    else:
        factor = 1.
        print("Did not get what unit you want, will memory return in bytes")
    return stats.available * factor


def lag_finder(y1, y2, Fs):
    """TODO

    Args:
        y1 ([type]): [description]
        y2 ([type]): [description]
        Fs ([type]): [description]

    Returns:
        [type]: [description]
    """
    n = len(y1)

    corr = signal.correlate(y2, y1, mode='same') / np.sqrt(signal.correlate(
        y1, y1, mode='same')[int(n/2)] * signal.correlate(y2, y2, mode='same')[int(n/2)])

    delay_arr = np.linspace(-0.5*n/Fs, 0.5*n/Fs, n)
    delay = delay_arr[np.argmax(corr)]

    return int(delay*Fs)


def get_timing(spikes):
    "Return timing of spikes"
    return np.asarray([[i, spike] for i, spike in enumerate(spikes) if spike > 0]).T


def compression_eeg(x, comp_fact=1):
    """TODO

    Args:
        x ([type]): [description]
        comp_fact (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    sign = np.sign(x)
    value = np.abs(x)**comp_fact
    return np.multiply(sign, value)


def create_events(dirac):
    """TODO

    Args:
        dirac ([type]): [description]

    Returns:
        [type]: [description]
    """
    timing = get_timing(dirac)
    events = np.zeros([len(timing), 3])
    events[:, 2] += 1
    events[:, 0] = timing
    return events.astype(int)

def center_weight(X,weight):
    """
    Center and weight the data
    Args:
        X (list): list of numpy arrays
        weights (list): list 
    """
    meanX = [Xk.mean(0,keepdims=True) for Xk in  X]
    X = [(Xk-mx)/w for Xk,mx,w in zip(X,meanX,weight)]
    return X, meanX

def count_significant_figures(num):
    if num == 0:
        return 0
    s = f"{num:.15g}"  # Convert the number to a string using general format with precision
    if 'e' in s:  # Handle scientific notation
        s = f"{float(s):f}"  # Convert back to float and then to normal fixed-point notation
    # Remove leading zeros and decimal points
    s = s.strip("0").replace(".", "")
    return len(s)
