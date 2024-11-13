"""
Common helper functions for modelling.
"""

import os
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from mne.decoding import BaseEstimator
from ..utils import lag_matrix, lag_span, lag_sparse, mem_check, get_timing
from ..viz import get_spatial_colors
from scipy import linalg
from scipy.stats import spearmanr
import mne
import itertools
from time import time as chrono


def _get_covmat(x, y):
    '''
    Helper function for computing auto-correlation / covariance matrices.
    '''
    return np.dot(x.T, y)


def _corr_multifeat(yhat, ytrue, nchans):
    '''
    Helper functions for computing correlation coefficient (Pearson's r) for multiple channels at once.
    Parameters
    ----------
    yhat : ndarray (T x nchan), estimate
    ytrue : ndarray (T x nchan), reference
    nchans : number of channels
    Returns
    -------
    corr_coeffs : 1-D vector (nchan), correlation coefficient for each channel
    '''
    return np.diag(np.corrcoef(x=yhat, y=ytrue, rowvar=False), k=nchans)


def _rankcorr_multifeat(yhat, ytrue, nchans):
    '''
    Helper functions for computing rank correlation coefficient (Spearman's r) for multiple channels at once.
    Parameters
    ----------
    yhat : ndarray (T x nchan), estimate
    ytrue : ndarray (T x nchan), reference
    nchans : number of channels
    Returns
    -------
    corr_coeffs : 1-D vector (nchan), correlation coefficient for each channel
    '''
    return np.diag(spearmanr(yhat, ytrue)[0], k=nchans)



def _rmse_multifeat(yhat, ytrue, axis=0):
    '''
    Helper functions for computing RMSE for multiple channels at once.
    Parameters
    ----------
    yhat : ndarray (T x nchan), estimate
    ytrue : ndarray (T x nchan), reference
    axis : axis to compute the RMSE along
    Returns
    -------
    rmses : 1-D vector (nchan), RMSE for each channel
    '''
    return np.sqrt(np.mean((yhat-ytrue)**2, axis))

def _r2_multifeat(yhat, ytrue, axis=0):
    '''
    Helper function for computing the coefficient of determination (R²) for multiple channels at once.
    Parameters
    ----------
    yhat : ndarray (T x nchan), estimate
    ytrue : ndarray (T x nchan), reference
    axis : axis along which to compute the R²
    Returns
    -------
    r2_scores : 1-D vector (nchan), R² for each channel
    '''
    ss_res = np.sum((ytrue - yhat) ** 2, axis=axis)  # Sum of squares of residuals
    ss_tot = np.sum((ytrue - np.mean(ytrue, axis=axis)) ** 2, axis=axis)  # Total sum of squares
    r2_scores = 1 - (ss_res / ss_tot)  # R² score for each channel
    return r2_scores

def _ezr2_multifeat(yhat, ytrue, Xtest, window_length, from_cov = False, axis = 0):
    ss_res = np.sum((ytrue - yhat) ** 2, axis=axis)  # Sum of squares of residuals
    ss_tot = np.sum((ytrue - np.mean(ytrue, axis=axis)) ** 2, axis=axis)  # Total sum of squares
    r2_scores = 1 - (ss_res / ss_tot)  # R² score for each channel

    n = ytrue.shape[0]
    p = Xtest.shape[1] * window_length
    r2_adjusted = 1 - ((1 - r2_scores) * (n - 1) / (n - p - 1))

    return r2_adjusted 
    

def _adjr2_multifeat(yhat, ytrue, Xtrain, Xtest, alpha, lags, from_cov = False, axis = 0, drop = True):
    '''
    Helper function for computing the adjusted coefficient of determination (R²) for multiple channels at once.
    from Lage et.al 2024, https://www.biorxiv.org/content/10.1101/2024.03.04.583270v1.full.pdf+html.
    Code repurposed from: https://github.com/mlsttin/adjustingR2
    Parameters
    ----------
    yhat : ndarray (T x nchan), estimate
    ytrue : ndarray (T x nchan), reference
    Xtrain: ndarray (T x nfeat), feat matrix of training data
    Xtest: ndarray (T x nfeat), feat matrix of testing data
    alpha: a single regularization parameter
    lags: list of lags, generally provided in the TRF object
    axis : axis along which to compute the R²
    Returns
    -------
    adj_r2_scores : 1-D vector (nchan), R² for each channel
    '''
    ss_res = np.sum((ytrue - yhat) ** 2, axis=axis)  # Sum of squares of residuals
    ss_tot = np.sum((ytrue - np.mean(ytrue, axis=axis)) ** 2, axis=axis)  # Total sum of squares
    r2_scores = 1 - (ss_res / ss_tot)  # non-adjusted R² score for each channel

    Xtrain = lag_matrix(Xtrain, lag_samples=lags,
               drop_missing=drop, filling=np.nan if drop else 0.)
    Xtest = lag_matrix(Xtest, lag_samples=lags,
           drop_missing=drop, filling=np.nan if drop else 0.)
    
    ntrain = Xtrain.shape[0]
    ntest, p = Xtest.shape
    Cte = np.eye(ntest) - (1./ntest) * np.ones((ntest,1)) @ np.ones((1,ntest))
    # Compute covariance matrices
    XtX = _get_covmat(Xtrain, Xtrain)
    I = np.eye(XtX.shape[0])

    # Compute eigenvalues and eigenvectors of covariance matrix XtX
    S, V = linalg.eigh(XtX, overwrite_a=False)

    # Sort the eigenvalues
    s_ind = np.argsort(S)[::-1]
    S = S[s_ind]
    V = V[:, s_ind]

    # Pick eigenvalues close to zero, remove them and corresponding eigenvectors
    # and compute the average
    tol = np.finfo(float).eps
    r = sum(S > tol)
    S = S[:r]
    V = V[:, :r]
    nl = np.mean(S)

    # Compute H0 and K0
    Xplus = np.linalg.inv(XtX + nl*alpha*I)@Xtrain.T
    H0 = Xtest@Xplus
    Rtest = Xtest - H0@Xtrain
    K0 = (Xtest - H0@Xtrain)

    # Compute pessimistic term
    kpess = -np.linalg.norm(H0, 'fro')**2 / H0.shape[1]
    ksho = -np.linalg.norm(K0, 'fro')**2/K0.shape[1]/np.trace(Xtest.T@Cte@Xtest)
    
    adj_r2 = (r2_scores - kpess) / (1 - kpess + ksho)
    
    return adj_r2



def _ridge_fit_SVD(x, y, alpha=[0.], from_cov=False, alpha_feat = False, n_feat = 1):
    '''
    SVD-inspired fast implementation of the SVD fitting.
    Note: When fitting the intercept, it's also penalized!
          If on doesn't want that, simply use average for each channel of y to estimate intercept.

    Parameters
    ----------
    X : ndarray (nsamples x nfeats) or autocorrelation matrix XtX (nfeats x nfeats) (if from_cov == True)
    y : ndarray (nsamples x nchans) or covariance matrix XtY (nfeats x nchans) (if from_cov == True)
    alpha : array-like.
        Default: [0.].
        List of regularization parameters. Regularization is applied 
        If 1D -> range of regularization params for the model (same reg. for all coeffs.)
    from_cov : bool
        Default: False.
        Use covariance matrices XtX & XtY instead of raw x, y arrays.
    alpha_feat : bool
        Default: False.
        If True, regularization is applied per feature. In this case, alpha is being re-written as
        all the possible combinations of alpha. This exponentianates computation time, avoid if possible
        or reduce to a minimum the possible combinations
    Returns
    -------
    model_coef : ndarray (model_feats* x alphas) *-specific shape depends on the model

    TO DO: allows to input specific alpha matrices rather than computing all cominations.
    '''
    # Compute covariance matrices
    if not from_cov:
        XtX = _get_covmat(x, x)
        XtY = _get_covmat(x, y)
    else:
        XtX = x[:]
        XtY = y[:]

    # Cast alpha in ndarray
    if isinstance(alpha, float):
        alpha = np.asarray([alpha])
    else:
        alpha = np.asarray(alpha)

    # Compute eigenvalues and eigenvectors of covariance matrix XtX
    S, V = linalg.eigh(XtX, overwrite_a=False)

    # Sort the eigenvalues
    s_ind = np.argsort(S)[::-1]
    S = S[s_ind]
    V = V[:, s_ind]

    # Pick eigenvalues close to zero, remove them and corresponding eigenvectors
    # and compute the average
    tol = np.finfo(float).eps
    r = sum(S > tol)
    S = S[:r]
    V = V[:, :r]
    nl = np.mean(S)

    # If per-coefficient regularization sort and drop alphas as well
    if alpha_feat:
        combinations = list(itertools.product(alpha , repeat=n_feat))
        n_lag = XtX.shape[0] // n_feat
        new_alpha = np.zeros((len(combinations), XtX.shape[0]))
        for i, comb in enumerate(combinations):
            new_alpha[i, :] = np.repeat(comb, n_lag)
        new_alpha = new_alpha[:,s_ind] # Sort according to eigenvals
        new_alpha = new_alpha[:, :r] # Drop coefficients corresponding to 'zero' eigenvals

    # Compute z
    z = np.dot(V.T, XtY)

    # Initialize empty list to store coefficient for different regularization parameters
    coeff = []

    # Compute coefficients for different regularization parameters
    if alpha_feat:
        for l in new_alpha:
            coeff.append(np.dot(V, (z/(S + nl*l)[:, np.newaxis])))
    else:
        for l in alpha:
            coeff.append(np.dot(V, (z/(S + nl*l)[:, np.newaxis])))
    
    return np.stack(coeff, axis=-1)


def _objective_value(y,X,mu,B,lambdas0,lambda1):
    '''
    Computation of the error for a least square regression with the possibility for 2 types of regularization.
    ''' 
    #Calc 1/(2n)|Y-1*mu'-sum(Xi*Bi)|^2 + lam0/2*sum(|Bi|_F^2) + lam1*sum(|Bi|_*)
    n,q = y.shape
    K = len(X)
    obj = 0
    pred = np.ones((n,1))@mu.T
    for i in range(K):
        pred = pred+X[i]@B[i]
        obj = obj + lambdas0/2*linalg.norm(B[i],ord='fro')**2+lambda1*sum(linalg.svdvals(B[i]))
    obj = obj+(1/(2*n))*np.nansum((y-pred)**2)

    return obj

def _soft_threshold(d,lam):
    '''
    Soft thresholding function.
    d is the array of singular values
    lam is a positive threshold
    '''
    dout = d.copy()
    np.fmax(d-lam,0,where=d>0,out=dout)
    np.fmin(d+lam,0,where=d<0,out=dout)

    return dout