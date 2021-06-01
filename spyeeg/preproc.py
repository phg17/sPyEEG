#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:48:37 2020

@author: phg17
"""

# Libraries
# Standard library
import numpy as np
from scipy import signal
from scipy.linalg import eigh as geigh
from joblib import Parallel, delayed
from sklearn.covariance import oas, ledoit_wolf, fast_mcd, empirical_covariance

# My libraries

# Mapping different estimator on the sklearn toolbox


def _lwf(X):
    """Wrapper for sklearn ledoit wolf covariance estimator"""
    C, _ = ledoit_wolf(X)
    return C


def _oas(X):
    """Wrapper for sklearn oas covariance estimator"""
    C, _ = oas(X)
    return C


def _scm(X):
    """Wrapper for sklearn sample covariance estimator"""
    return empirical_covariance(X)


def _mcd(X):
    """Wrapper for sklearn mcd covariance estimator"""
    _, C, _, _ = fast_mcd(X)
    return C


def _cov(X):
    "Wrapper for numpy cov estimator"
    return np.cov(X, rowvar=False)


def _corr(X):
    "Wrapper for numpy cov estimator"
    return np.corrcoef(X, rowvar=False)


def _check_est(est):
    """Check if a given estimator is valid"""

    # Check estimator exist and return the correct function
    estimators = {
        'cov': _cov,
        'scm': _scm,
        'lwf': _lwf,
        'oas': _oas,
        'mcd': _mcd,
        'corr': _corr
    }

    if callable(est):
        # All good (cross your fingers)
        pass
    elif est in estimators.keys():
        # Map the corresponding estimator
        est = estimators[est]
    else:
        # raise an error
        raise ValueError(
            """%s is not an valid estimator ! Valid estimators are : %s or a
             callable function""" % (est, (' , ').join(estimators.keys())))
    return est


def covariances(X, estimator='cov'):
    """Estimation of covariance matrices."""
    est = _check_est(estimator)
    # -> Loosen this requirement (any n-dim vector?)? Or handle it elsewhere?
    assert X.ndim == 3, "Data must be 3d (trials, samples, channels)"
    Ntrials, Nsamples, Nchans = X.shape
    covmats = np.zeros((Ntrials, Nchans, Nchans))
    for i in range(Ntrials):
        covmats[i, :, :] = est(X[i, :, :])
    return covmats

def create_filterbank(freqs, srate, filtertype=signal.cheby2, **kwargs):
    """Creates a filter bank, by default of chebychev type 2 filters.
    Parameters of filter are to be defined as name value pair arguments.
    Frequency bands are defined with boundaries instead of center frequencies.
    """
    normalized_freqs = np.asarray(freqs)/(srate/2.) # create normalized frequencies specifications
    return [filtertype(**kwargs, Wn=ff) for ff in normalized_freqs]

def apply_filterbank(data, fbank, filt_func=signal.lfilter, n_jobs=-1, axis=-1):
    """Applies a filterbank to a given multi-channel signal.
    Parameters
    ----------
    data : ndarray (samples, nchannels)
    fb : list
        list of (b,a) tuples, where b and a specify a digital filter
    Returns
    -------
    y : ndarray (nfilters, samples, nchannels)
    """
    return np.asarray(Parallel(n_jobs=n_jobs)(delayed(filt_func)(b, a, data, axis=axis) for b, a in fbank))