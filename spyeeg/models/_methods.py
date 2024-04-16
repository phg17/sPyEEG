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
import mne
import itertools


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


def dirac_distance(dirac1, dirac2, Fs, window_size=0.01):
    '''
    Fast implementation of victor-purpura spike distance (faster than neo & elephant python packages)
    Direct Python port of http://www-users.med.cornell.edu/~jdvicto/pubalgor.htmlself.
    The below code was tested against the original implementation and yielded exact results.
    All credits go to the authors of the original code.
    Input:
        s1,2: pair of vectors of spike times
        cost: cost parameter for computing Victor-Purpura spike distance.
        (Note: the above need to have the same units!)
    Output:
        d: VP spike distance.
    '''
    cost = Fs * window_size
    s1 = get_timing(dirac1)
    s2 = get_timing(dirac2)

    nspi = len(s1)
    nspj = len(s2)

    scr = np.zeros((nspi+1, nspj+1))

    scr[:, 0] = np.arange(nspi+1)
    scr[0, :] = np.arange(nspj+1)

    for i in np.arange(1, nspi+1):
        for j in np.arange(1, nspj+1):
            scr[i, j] = min([scr[i-1, j]+1, scr[i, j-1]+1,
                             scr[i-1, j-1]+cost*np.abs(s1[i-1]-s2[j-1])])

    d = scr[nspi, nspj]

    return d


def _Dlp(A, B, p=2):
    cost = np.sum(np.power(np.abs(A - B), p))
    return np.power(cost, 1 / p)


def _twed(A, timeSA, B, timeSB, nu, _lambda):
    # [distance, DP] = TWED( A, timeSA, B, timeSB, lambda, nu )
    # Compute Time Warp Edit Distance (TWED) for given time series A and B
    #
    # A      := Time series A (e.g. [ 10 2 30 4])
    # timeSA := Time stamp of time series A (e.g. 1:4)
    # B      := Time series B
    # timeSB := Time stamp of time series B
    # lambda := Penalty for deletion operation
    # nu     := Elasticity parameter - nu >=0 needed for distance measure
    # Reference :
    #    Marteau, P.; F. (2009). "Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching".
    #    IEEE Transactions on Pattern Analysis and Machine Intelligence. 31 (2): 306–318. arXiv:cs/0703033
    #    http://people.irisa.fr/Pierre-Francois.Marteau/

    # Check if input arguments
    if len(A) != len(timeSA):
        print("The length of A is not equal length of timeSA")
        return None, None

    if len(B) != len(timeSB):
        print("The length of B is not equal length of timeSB")
        return None, None

    if nu < 0:
        print("nu is negative")
        return None, None

    # Add padding
    A = np.array([0] + list(A))
    timeSA = np.array([0] + list(timeSA))
    B = np.array([0] + list(B))
    timeSB = np.array([0] + list(timeSB))

    n = len(A)
    m = len(B)
    # Dynamical programming
    DP = np.zeros((n, m))

    # Initialize DP Matrix and set first row and column to infinity
    DP[0, :] = np.inf
    DP[:, 0] = np.inf
    DP[0, 0] = 0

    # Compute minimal cost
    for i in range(1, n):
        for j in range(1, m):
            # Calculate and save cost of various operations
            C = np.ones((3, 1)) * np.inf
            # Deletion in A
            C[0] = (
                DP[i - 1, j]
                + _Dlp(A[i - 1], A[i])
                + nu * (timeSA[i] - timeSA[i - 1])
                + _lambda
            )
            # Deletion in B
            C[1] = (
                DP[i, j - 1]
                + _Dlp(B[j - 1], B[j])
                + nu * (timeSB[j] - timeSB[j - 1])
                + _lambda
            )
            # Keep data points in both time series
            C[2] = (
                DP[i - 1, j - 1]
                + _Dlp(A[i], B[j])
                + _Dlp(A[i - 1], B[j - 1])
                + nu * (abs(timeSA[i] - timeSB[j]) +
                        abs(timeSA[i - 1] - timeSB[j - 1]))
            )
            # Choose the operation with the minimal cost and update DP Matrix
            DP[i, j] = np.min(C)
    distance = DP[n - 1, m - 1]
    return distance, DP


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
    S, V = linalg.eigh(XtX, overwrite_a=False, turbo=True)

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