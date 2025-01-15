"""
Common helper functions for modelling.
"""

import os
import numpy as np
import itertools
from sklearn.linear_model import RidgeCV, LinearRegression
import matplotlib.pyplot as plt
import mne
from scipy import linalg
from scipy.stats import spearmanr, pearsonr
from scipy.linalg import sqrtm
from scipy.signal.windows import get_window
from scipy.fft import fft, ifft
from scipy.signal import csd as welch_csd, fftconvolve, welch
from ..utils import lag_matrix, lag_span, lag_sparse, mem_check, get_timing


def _get_covmat(x, y):
    """
    Helper function for computing auto-correlation / covariance matrices.

    Parameters
    ----------
    x : ndarray
    y : ndarray

    Returns
    ----------
    covmat : covariance matrix
    """
    return np.dot(x.T, y)


def _corr_multifeat(yhat, ytrue, nchans):
    """
    Helper functions for computing correlation coefficient (Pearson's r) for multiple channels at once.
    
    Parameters
    ----------
    yhat : ndarray
        estimate, of shape (T x nchan)
    ytrue : ndarray
        reference, of shape (T x nchan)
    nchans : int
        number of channels.
    
    Returns
    ----------
    corr_coeffs : ndarray
        1-D vector correlation coefficient for each channel, of shape (nchans,).
    """
    return np.diag(np.corrcoef(x=yhat, y=ytrue, rowvar=False), k=nchans)


def _rankcorr_multifeat(yhat, ytrue, nchans):
    """
    Helper functions for computing rank correlation coefficient (Spearman's r) for multiple channels at once.
    
    Parameters
    ----------
    yhat : ndarray
        estimate, of shape (T x nchan)
    ytrue : ndarray
        reference, of shape (T x nchan)
    nchans : int
        number of channels.
    
    Returns
    ----------
    corr_coeffs : ndarray
        1-D vector correlation coefficient for each channel, of shape (nchans).
    """
    return np.diag(spearmanr(yhat, ytrue)[0], k=nchans)



def _rmse_multifeat(yhat, ytrue, axis=0):
    """
    Helper functions for computing RMSE for multiple channels at once.
    
    Parameters
    ----------
    yhat : ndarray
        estimate, of shape (T x nchan)
    ytrue : ndarray
        reference, of shape (T x nchan)
    axis : int
        axis to compute the RMSE along
    
    Returns
    ----------
    rmses : ndarray
        1-D vector, RMSE for each channel, of shape (nchan,)
    """
    return np.sqrt(np.mean((yhat-ytrue)**2, axis))
    

def _r2_multifeat(yhat, ytrue, axis=0):
    """
    Helper function for computing the coefficient of determination (R²) for multiple channels at once.
    
    Parameters
    ----------
    yhat : ndarray
        estimate, of shape (T x nchan)
    ytrue : ndarray
        reference, of shape (T x nchan)
    axis : int
        axis to compute the RMSE along
    
    Returns
    ----------
    r2_scores : ndarray
        1-D vector of R² for each channel, of shape (nchan,)
    """
    ss_res = np.sum((ytrue - yhat) ** 2, axis=axis)  # Sum of squares of residuals
    ss_tot = np.sum((ytrue - np.mean(ytrue, axis=axis)) ** 2, axis=axis)  # Total sum of squares
    r2_scores = 1 - (ss_res / ss_tot)  # R² score for each channel
    return r2_scores

def _ezr2_multifeat(yhat, ytrue, Xtest, window_length, from_cov = False, axis = 0):
    """
    Helper function for computing Ezekiel correction for the coefficient of determination (R²) for multiple channels at once.
    
    Parameters
    ----------
    yhat : ndarray
        estimate, of shape (T x nchan)
    ytrue : ndarray
        reference, of shape (T x nchan)
    axis : int
        axis to compute the RMSE along
    
    Returns
    ----------
    r2_scores : ndarray
        1-D vector of Ezekiel corrected R² for each channel, of shape (nchan,)
    """
    ss_res = np.sum((ytrue - yhat) ** 2, axis=axis)  # Sum of squares of residuals
    ss_tot = np.sum((ytrue - np.mean(ytrue, axis=axis)) ** 2, axis=axis)  # Total sum of squares
    r2_scores = 1 - (ss_res / ss_tot)  # R² score for each channel

    n = ytrue.shape[0]
    p = Xtest.shape[1] * window_length
    r2_adjusted = 1 - ((1 - r2_scores) * (n - 1) / (n - p - 1))

    return r2_adjusted 
    

def _adjr2_multifeat(yhat, ytrue, Xtrain, Xtest, alpha, lags, from_cov = False, axis = 0, drop = True):
    """
    Helper function for computing the adjusted coefficient of determination (R²) for multiple channels at once.
    from Lage et.al 2024, https://www.biorxiv.org/content/10.1101/2024.03.04.583270v1.full.pdf+html.
    Code repurposed from: https://github.com/mlsttin/adjustingR2
    
    Parameters
    ----------
    yhat : ndarray
        Estimate array of shape (T x nchan).
    ytrue : ndarray
        Reference array of shape (T x nchan).        
    Xtrain: ndarray
        Feature matrix of training data, of shape (T x nfeat).
    Xtest: ndarray, T x nfeat
        Feature matrix of testing data, of shape (T x nfeat).
    alpha: float
        A single regularization parameter.
    lags: list
        A list of lags, generally provided in the TRF object.
    axis : int
        axis along which to compute the R²
    
    Returns
    -------
    adj_r2_scores : ndarray
        1-D vector of corrected R² for each channel, of shape (nchan,)
    """
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

def _pairwise_corr(X, Y):
    """
    Function for computing the pairwise correlations between the columns of two matrices X and Y

    Parameters
    ----------
    X: ndarray
        estimate array of shape (T x nfeat)
    Y: ndarray
        reference array of shape (T x nfeat)

    Returns
    -------
    correlations: ndarray 
        1D vector of correlation, of shape (n_features,)
    """
    
    return np.array([pearsonr(X.real[:, i], Y.real[:, i])[0] for i in range(X.shape[1])])


def _ridge_fit_SVD(x, y, alpha=[0.], from_cov=False, alpha_feat = False, n_feat = 1):
    '''
    SVD-inspired fast implementation of the SVD fitting.

    Parameters
    ----------
    x : ndarray
        - if from_cov == False (default): Feature matrix X (nsamples x nfeats) 
        - if from_cov == True: Autocorrelation matrix XtX (nfeats x nfeats) 
    y : ndarray 
        - if from_cov == False (default): Target matrix Y (nsamples x nchans) 
        - if from_cov == True: Covariance matrix XtY (nfeats x nchans) 
    alpha : array-like
        Default: [0.] i.e. no regularization
        List of regularization parameters. 
    from_cov : bool
        Use covariance matrices XtX & XtY instead of raw x, y arrays.
    alpha_feat : bool
        If True, regularization is applied per feature. 
        All possible combinations of alpha are tested, which exponentianates computation time, avoid in most cases.
    Returns
    -------
    model_coef : ndarray 
        Coefficients of the Ridge, specific shape depends on the model.
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


def _fourier_fit(x, y, alpha=[0.], lags = [-1,1]):
    """
    Estimate the IRF in the frequency domain using FFT.

    Parameters
    ----------
    x : ndarray
        Feature matrix x (nsamples x nfeats) 
    y : ndarray 
        Target matrix y (nsamples x nchans) 
    alpha : array-like
        Default: [0.] i.e. no regularization
        List of regularization parameters. 
    lags : array-like
        Default: [-1,1]
        List of lags to consider.
    Returns
    -------
    model_coef : ndarray 
        Coefficients of the Ridge, specific shape depends on the model.
    """
    n_samples, n_features = x.shape
    n_samples, n_outputs = y.shape
    min_lag, max_lag = lags.min(), lags.max()
    total_lags = max_lag - min_lag + 1
    x_padded = np.pad(x, ((0, total_lags), (0, 0)), mode='constant')
    y_padded = np.pad(y, ((0, total_lags), (0, 0)), mode='constant')

    X_fft = fft(x_padded, axis=0)  # Shape: (n_samples + total_lags, n_features)
    Y_fft = fft(y_padded, axis=0)  # Shape: (n_samples + total_lags, n_outputs)

    S_xy = X_fft[:, :, None] @ Y_fft[:, None, :].conjugate()
    S_xx = X_fft[:, :, None] @ X_fft[:, None, :].conjugate()

    S_xy = _resample_array(S_xy, total_lags*2)
    S_xx = _resample_array(S_xx, total_lags*2)

    window = get_window('boxcar', 2)
    S_xy = np.apply_along_axis(lambda m: fftconvolve(m, window, mode='same'), axis=0, arr=S_xy)
    S_xx = np.apply_along_axis(lambda m: fftconvolve(m, window, mode='same'), axis=0, arr=S_xx)
    
    irf = np.zeros((total_lags, n_features, n_outputs, len(alpha)))

    for a_index, a in enumerate(alpha):
        reg_matrix = a * np.eye(S_xx.shape[-1])[None, :, :] * np.diag(np.mean(S_xx, axis=0).real).mean()
        H_fft = np.linalg.inv(S_xx + reg_matrix) @ S_xy

        # Convert transfer function back to time domain
        H_time = np.real(ifft(H_fft, axis=0))  # Time-domain transfer function
    
        # Align IRF to lag range
        for f in range(n_features):
            for o in range(n_outputs):
                irf[:, f, o, a_index] = np.roll(H_time[:, f, o], max_lag)[:total_lags]

    return np.vstack(irf)

def _b2b(t,X1,X2,Y1,Y2, alphax, alphay, Ridge2 = True):
    """
    Back to back regression fitting.

    Parameters
    ----------
    t : int
        sample index to consider
    X1 : ndarray
    """
    y1 = Y1[:,t,:]
    y2 = Y2[:,t,:]

    #predict each feature Xi from all channels Y (i.e. decoding)
    reg1 = RidgeCV(alphas=alphay, fit_intercept=False, cv = None, scoring = 'neg_mean_squared_error') 
    reg1.fit(y1, X1)
    G = reg1.coef_.T

    # reg2 = LinearRegression(fit_intercept=False) #King et al., 2020
    reg2 = RidgeCV(alphas=alphax, fit_intercept=False, cv = None, scoring = 'neg_mean_squared_error') #Gwilliams et al., 2024
    reg2.fit(X2, np.dot(y2, G))
    H = reg2.coef_.T

    #return causal influence matrix
    return H.diagonal()


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

def _covariance_fourier(x, start_lag, end_lag):
    """
    Compute the covariance matrix for lagged multi-channel data using the Fourier method.

    Parameters
    ----------
    x : numpy array of shape (n_samples, n_channels)
        Input time series data with multiple channels.
    start_lag : int
        Start of the lag range.
    end_lag : int
        End of the lag range.

    Returns
    -------
    cov_X : numpy array of shape (n_channels * n_lags, n_channels * n_lags)
            Covariance matrix for the lagged data across all channels.
    """
    n_samples, n_channels = x.shape
    lags = np.arange(start_lag, end_lag)
    n_lags = len(lags)

    # Initialize the covariance matrix
    cov_X = np.zeros((n_channels * n_lags, n_channels * n_lags))

    # Compute FFT for each channel
    fft_x = [np.fft.fft(x[:, i], n=2 * n_samples) for i in range(n_channels)]
    power_spectra = [np.abs(fft) ** 2 for fft in fft_x]

    # Compute cross-power spectra for all pairs of channels
    cross_power_spectra = {
        (i, j): fft_x[i] * np.conj(fft_x[j])
        for i in range(n_channels) for j in range(i, n_channels)
    }

    # Compute autocorrelation and cross-correlation via inverse FFT
    autocorrs = [
        np.fft.ifft(power_spectra[i]).real[:n_samples] for i in range(n_channels)
    ]
    crosscorrs = {
        (i, j): np.fft.ifft(cross_power_spectra[(i, j)]).real[:n_samples]
        for i in range(n_channels) for j in range(i, n_channels)
    }

    # Fill the covariance matrix
    for i, lag1 in enumerate(lags):
        for j, lag2 in enumerate(lags):
            lag_diff = abs(lag1 - lag2)

            for c1 in range(n_channels):
                for c2 in range(n_channels):
                    block_i = c1 * n_lags + i
                    block_j = c2 * n_lags + j

                    if c1 == c2:  # Autocorrelation
                        cov_X[block_i, block_j] = (
                            autocorrs[c1][lag_diff] if lag_diff < n_samples else 0
                        )
                    elif c1 < c2:  # Cross-correlation (upper triangle)
                        cov_X[block_i, block_j] = (
                            crosscorrs[(c1, c2)][lag_diff] if lag_diff < n_samples else 0
                        )
                    else:  # Cross-correlation (lower triangle)
                        cov_X[block_i, block_j] = (
                            crosscorrs[(c2, c1)][lag_diff] if lag_diff < n_samples else 0
                        )

    return cov_X

def _inverse_square_root(m):

    '''
    Function for computing the inverse square-root of a function. Multiple methods are available, but for a quick implementation
    here I just use the scipy.linalg.sqrtm function.

    Parameters
    ----------
    m: numpy array of shape (n_features, n_features)
    '''

    return np.linalg.inv(sqrtm(m))


def _resample_array(array, new_length):
    """
    Resamples an array to a fixed number of points using average pooling.

    Parameters
    ----------
        array (np.ndarray): The input array to resample. Can be 1D or multi-channel (e.g., 2D for multi-channel).
        new_length (int): The desired length of the output array.

    Returns
    ----------
        np.ndarray: The resampled array.
    """
    if new_length <= 0:
        raise ValueError("new_length must be greater than 0.")

    # Handle multi-channel arrays
    if array.ndim == 1:
        array = array[:, np.newaxis]

    # Original array length
    old_length = array.shape[0]

    if old_length == new_length:
        return array.copy() if array.ndim == 1 else array.copy().squeeze()

    # Compute the resampling ratio
    ratio = old_length / new_length

    # Create new indices
    new_indices = np.linspace(0, old_length, new_length, endpoint=False)

    # Initialize the resampled array
    resampled = np.zeros((new_length, *[array.shape[i] for i in range(1, len(array.shape))]), dtype=array.dtype)

    # Average pooling for the segments
    for i in range(new_length):
        # Determine the range of indices in the original array that contribute to this output index
        start_idx = int(np.floor(i * ratio))
        end_idx = int(np.ceil((i + 1) * ratio))

        # If the range is within bounds, compute the mean
        if end_idx > start_idx:
            resampled[i] = np.mean(array[start_idx:end_idx], axis=0)
        else:
            # Interpolate for fractional indices
            lower_idx = min(start_idx, old_length - 1)
            upper_idx = min(end_idx, old_length - 1)
            weight = (i * ratio - lower_idx)
            resampled[i] = (1 - weight) * array[lower_idx] + weight * array[upper_idx]

    return resampled

