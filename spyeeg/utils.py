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
    
    Parameters
    ----------
    audio: ndarray
        sound (PCM int)
        
    Returns
    -------
    audio: ndarray
        sound (PCM float)
    """
    iinfo = np.iinfo(audio.dtype)
    max_val = max(abs(iinfo.min), abs(iinfo.max))
    audio = audio/max_val
    return audio

def RMS(x):
    return np.sqrt(np.mean(np.power(x, 2)))


def lag_matrix(data, lag_samples=(-1, 0, 1), filling=np.nan, drop_missing=False):
    """Helper function to create a matrix of lagged time series.

    The lag can be arbitrarily spaced. Check other functions to create series of lags
    whether they are contiguous or sparsely spanning a time window :func:`lag_span` and
    :func:`lag_sparse`.

    Parameters
    ----------
    data : ndarray 
        Multivariate data, of shape (nsamples, nfeats).
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
    lagged : ndarray 
        Matrix of lagged time series, of shape (nsamples_new, nfeats*len(lag_samples))

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


def lag_span(tmin, tmax, srate=100):
    """Create an array of lags spanning the time window [tmin, tmax].

    Parameters
    ----------
    tmin : float
        In seconds.
    tmax : float
        In seconds.
    srate : float
        Sampling rate.

    Returns
    -------
    lags : 1darray
        Array of lags in _samples_

    """
    sample_min, sample_max = int(
        np.ceil(tmin * srate)), int(np.ceil(tmax * srate))
    return np.arange(sample_min, sample_max)


def lag_sparse(times, srate=100):
    """Create an array of lags for the requested time point in `times`.

    Parameters
    ----------
    times : list
        List of time point in seconds.
    srate : float
        Sampling rate.

    Returns
    -------
    lags : 1darray
        Array of lags in _samples_

    """
    return np.asarray([int(np.ceil(t * srate)) for t in times])


def _is_1d(arr):
    """Short utility function to check if an array is vector-like"""
    return np.product(arr.shape) == max(arr.shape)


def is_pos_def(A):
    """Check if matrix is positive definite (see https://stackoverflow.com/a/44287862/5303618)
    """
    if np.array_equal(A, A.conj().T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


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


def get_timing(spikes):
    "Return timing of spikes"
    return np.asarray([[i, spike] for i, spike in enumerate(spikes) if spike != 0]).T

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








