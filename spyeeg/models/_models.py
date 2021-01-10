"""
Original long module.
"""

#import logging
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from mne.decoding import BaseEstimator
from ..utils import lag_matrix, lag_span, lag_sparse, mem_check, get_timing
from ..viz import get_spatial_colors
from scipy import linalg
#import elephant
#import dtw
import mne
#import pyriemann

# logging.getLogger('matplotlib').setLevel(logging.WARNING)
# logging.basicConfig(level=logging.WARNING)
#LOGGER = logging.getLogger(__name__.split('.')[0])


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


def Dlp(A, B, p=2):
    cost = np.sum(np.power(np.abs(A - B), p))
    return np.power(cost, 1 / p)


def twed(A, timeSA, B, timeSB, nu, _lambda):
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
                + Dlp(A[i - 1], A[i])
                + nu * (timeSA[i] - timeSA[i - 1])
                + _lambda
            )
            # Deletion in B
            C[1] = (
                DP[i, j - 1]
                + Dlp(B[j - 1], B[j])
                + nu * (timeSB[j] - timeSB[j - 1])
                + _lambda
            )
            # Keep data points in both time series
            C[2] = (
                DP[i - 1, j - 1]
                + Dlp(A[i], B[j])
                + Dlp(A[i - 1], B[j - 1])
                + nu * (abs(timeSA[i] - timeSB[j]) +
                        abs(timeSA[i - 1] - timeSB[j - 1]))
            )
            # Choose the operation with the minimal cost and update DP Matrix
            DP[i, j] = np.min(C)
    distance = DP[n - 1, m - 1]
    return distance, DP


def _ridge_fit_SVD(x, y, alpha=[0.], from_cov=False, alpha_feature=False):
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
        List of regularization parameters.
    from_cov : bool
        Default: False.
        Use covariance matrices XtX & XtY instead of raw x, y arrays.
    Returns
    -------
    model_coef : ndarray (model_feats* x alphas) *-specific shape depends on the model
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
    elif alpha_feature:
        alpha = np.asarray(alpha).T
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
    S = S[0:r]
    V = V[:, 0:r]
    nl = np.mean(S)

    # Compute z
    z = np.dot(V.T, XtY)

    # Initialize empty list to store coefficient for different regularization parameters
    coeff = []

    # Compute coefficients for different regularization parameters
    if alpha_feature:
        for l in alpha:
            coeff.append(np.dot(V, (z/(S + nl*l)[:, np.newaxis])))

    else:
        for l in alpha:
            coeff.append(np.dot(V, (z/(S[:, np.newaxis] + nl*l))))

    return np.stack(coeff, axis=-1)


class ERP_class():
    def __init__(self, tmin, tmax, srate, n_chan=63):
        self.srate = srate
        self.tmin = tmin
        self.tmax = tmax
        self.window = lag_span(tmin, tmax, srate)
        self.times = self.window/srate
        self.ERP = np.zeros(len(self.window))
        self.mERP = np.zeros([len(self.window), n_chan])
        self.single_events = []
        self.single_events_mult = []
        self.peak_time = None
        self.peak_arg = None

    def add_data(self, eeg, events, event_type='spikes'):

        if event_type == 'spikes':
            events_list = get_timing(events)
        else:
            events_list = events

        # for i in np.where(events_list < eeg.shape[0] - self.window[-1])[0]:
        for i in range(len(events_list)):
            try:
                event = events_list[i]
                self.ERP += np.sum((eeg[self.window + event]), axis=1)
                self.mERP += eeg[self.window + event, :]
                self.single_events.append(
                    np.sum((eeg[self.window + event]), axis=1))
            except:
                print('out of window')
        self.peak_time = np.argmax(self.ERP) / self.srate + self.tmin
        self.peak_arg = np.argmax(self.ERP) + self.tmin * self.srate

    def weight_data(self, eeg, cont_stim):

        # for i in np.where(events_list < eeg.shape[0] - self.window[-1])[0]:
        for i in range(len(cont_stim)):
            try:
                w = cont_stim[i]
                self.ERP += np.sum((w * eeg[self.window + i]), axis=1)
                self.mERP += w * eeg[self.window + i, :]
                self.single_events.append(
                    np.sum(np.abs(w * eeg[self.window + i]), axis=1))
                self.single_events_mult.append((w * eeg[self.window + i]))
            except:
                pass

    def inverse_weight_data(self, eeg, cont_stim):

        for n_chan in range(63):
            for t in range(len(cont_stim)):
                try:
                    w = eeg[t, n_chan]
                    self.mERP[:, n_chan] += w * cont_stim[self.window + t, 0]
                except:
                    pass
        self.ERP = np.sum(self.mERP, axis=1)

    def plot_simple(self):
        plt.figure()
        plt.plot(self.times, self.ERP)

    def plot_multi(self):
        plt.figure()
        plt.plot(self.times, self.mERP)

    def plot_topo(self, raw_info, Fs, time=None):

        f, (ax1, ax2) = plt.subplots(1, 2)
        f.set_figwidth(10)
        if not time:
            t = [np.argmin(self.ERP), np.argmax(self.ERP)]
        else:
            t = [int((time[0] - self.tmin)*Fs), int((time[1] - self.tmin)*Fs)]

        # Visualize topography max
        mne.viz.plot_topomap(
            self.mERP[t[0], :], raw_info, axes=ax1, cmap='RdBu_r', show=False)
        ax1.set_title('tlag={} ms'.format(
            int((t[0]/self.srate + self.tmin)*1000)))

        # Visualize topography min
        mne.viz.plot_topomap(
            self.mERP[t[1], :], raw_info, axes=ax2, cmap='RdBu_r', show=False)
        ax2.set_title('tlag={} ms'.format(
            int((t[1]/self.srate + self.tmin)*1000)))


class TRFEstimator(BaseEstimator):

    def __init__(self, times=(0.,), tmin=None, tmax=None, srate=1., alpha=[0.], fit_intercept=False, mtype='forward'):

        # Times reflect mismatch a -> b, where a - dependent, b - predicted
        # Negative timelags indicate a lagging behind b
        # Positive timelags indicate b lagging behind a
        # For example:
        # eeg -> env (tmin = -0.5, tmax = 0.1)
        # Indicate timeframe from -100 ms (eeg precedes stimulus): 500 ms (eeg after stimulus)
        self.tmin = tmin
        self.tmax = tmax
        self.times = times
        self.srate = srate
        self.alpha = alpha
        # Forward or backward. Required for formatting coefficients in get_coef (convention: forward - stimulus -> eeg, backward - eeg - stimulus)
        self.mtype = mtype
        self.fit_intercept = fit_intercept
        self.fitted = False
        self.lags = None

        # All following attributes are only defined once fitted (hence the "_" suffix)
        self.intercept_ = None
        self.coef_ = None
        self.n_feats_ = None
        self.n_chans_ = None
        self.feat_names_ = None
        self.valid_samples_ = None
        # Autocorrelation matrix of feature X (thus XtX) -> used for computing model using fit_from_cov
        self.XtX_ = None
        # Covariance matrix of features X and Y (thus XtX) -> used for computing model using fit_from_cov
        self.XtY_ = None

    def fill_lags(self):
        """Fill the lags attributes, with number of samples and times in seconds.
        Note
        ----
        Necessary to call this function if one wishes to use trf.lags _before_
        :func:`trf.fit` is called.

        """
        if self.tmin and self.tmax:
            # LOGGER.info("Will use lags spanning form tmin to tmax.\nTo use individual lags, use the `times` argument...")
            self.lags = lag_span(self.tmin, self.tmax, srate=self.srate)[
                ::-1]  # pylint: disable=invalid-unary-operand-type
            # self.lags = lag_span(-tmax, -tmin, srate=srate) #pylint: disable=invalid-unary-operand-type
            self.times = self.lags[::-1] / self.srate
        else:
            self.times = np.asarray(self.times)
            self.lags = lag_sparse(self.times, self.srate)[::-1]

    def get_XY(self, X, y, lagged=False, drop=True, feat_names=()):
        '''
        Preprocess X and y before fitting (finding mapping between X -> y)
        Parameters
        ----------
        X : ndarray (T x nfeat)
        y : ndarray (T x nchan)
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        feat_names : list
            Names of features being fitted. Must be of length ``nfeats``.
        Returns
        -------
        Features preprocessed for fitting the model.
        X : ndarray (T x nlags * nfeats)
        y : ndarray (T x nchan)
        '''
        self.fill_lags()

        X = np.asarray(X)
        y = np.asarray(y)

        # Estimate the necessary size to compute stuff
        y_memory = sum([yy.nbytes for yy in y]) if np.ndim(
            y) == 3 else y.nbytes
        estimated_mem_usage = X.nbytes * \
            (len(self.lags) if not lagged else 1) + y_memory
        if estimated_mem_usage/1024.**3 > mem_check():
            raise MemoryError("Not enough RAM available! (needed %.1fGB, but only %.1fGB available)" % (
                estimated_mem_usage/1024.**3, mem_check()))

        # Fill n_feat and n_chan attributes
        # if X has been lagged, divide by number of lags
        self.n_feats_ = X.shape[1] if not lagged else X.shape[1] // len(
            self.lags)
        self.n_chans_ = y.shape[1] if y.ndim == 2 else y.shape[2]

        # Assess if feat names corresponds to feat number
        if feat_names:
            err_msg = "Length of feature names does not match number of columns from feature matrix"
            if lagged:
                assert len(feat_names) == X.shape[1] // len(self.lags), err_msg
            else:
                assert len(feat_names) == X.shape[1], err_msg
            self.feat_names_ = feat_names

        # this include non-valid samples for now
        n_samples_all = y.shape[0] if y.ndim == 2 else y.shape[1]

        # drop samples that can't be reconstructed because on the edge, all is true otherwise
        if drop:
            self.valid_samples_ = np.logical_not(np.logical_or(np.arange(n_samples_all) < abs(max(self.lags)),
                                                               np.arange(n_samples_all)[::-1] < abs(min(self.lags))))
        else:
            self.valid_samples_ = np.ones((n_samples_all,), dtype=bool)

        # Creating lag-matrix droping NaN values if necessary
        y = y[self.valid_samples_, :] if y.ndim == 2 else y[:,
                                                            self.valid_samples_, :]
        if not lagged:
            X = lag_matrix(X, lag_samples=self.lags,
                           drop_missing=drop, filling=np.nan if drop else 0.)

        return X, y

    def fit(self, X, y, lagged=False, drop=True, feat_names=()):
        """Fit the TRF model.
        Mapping X -> y. Note the convention of timelags and type of model for seamless recovery of coefficients.
        Parameters
        ----------
        X : ndarray (nsamples x nfeats)
        y : ndarray (nsamples x nchans)
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        feat_names : list
            Names of features being fitted. Must be of length ``nfeats``.
        Returns
        -------
        coef_ : ndarray (alphas x nlags x nfeats)
        intercept_ : ndarray (nfeats x 1)
        """

        # Preprocess and lag inputs
        X, y = self.get_XY(X, y, lagged, drop, feat_names)

        # Adding intercept feature:
        if self.fit_intercept:
            X = np.hstack([np.ones((len(X), 1)), X])

        # Regress with Ridge to obtain coef for the input alpha
        self.coef_ = _ridge_fit_SVD(X, y, self.alpha)

        # Reshaping and getting coefficients
        if self.fit_intercept:
            self.intercept_ = self.coef_[0, np.newaxis, :]
            self.coef_ = self.coef_[1:, :]

        self.fitted = True

        return self

    def get_coef(self):
        '''
        Format and return coefficients. Note mtype attribute needs to be declared in the __init__.

        Returns
        -------
        coef_ : ndarray (nlags x nfeats x nchans x regularization params)
        '''
        if np.ndim(self.alpha) == 0:
            betas = np.reshape(self.coef_, (len(self.lags),
                                            self.n_feats_, self.n_chans_))
        else:
            betas = np.reshape(self.coef_, (len(self.lags),
                                            self.n_feats_, self.n_chans_, len(self.alpha)))

        if self.mtype == 'forward':
            betas = betas[::-1, :]

        return betas

    def add_cov(self, X, y, lagged=False, drop=True, n_parts=1):
        '''
        Compute and add (with normalization factor) covariance matrices XtX, XtY
        For v. large population models when it's not possible to load all the data to memory at once.
        Parameters
        ----------
        X : ndarray (nsamples x nfeats) or list/tuple of ndarray (from which the model will be computed)
        y : ndarray (nsamples x nchans) or list/tuple of ndarray (from which the model will be computed)
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        n_parts : number of parts from which the covariance matrix are computed (required for normalization)
            Default: 1
        Returns
        -------
        XtX : autocorrelation matrix for X (accumulated)
        XtY : covariance matrix for X & Y (accumulated)
        '''
        if isinstance(X, (list, tuple)) and n_parts > 1:

            assert len(X) == len(y)

            for part in range(len(X)):
                assert len(X[part]) == len(y[part])
                X_part, y_part = self.get_XY(X[part], y[part], lagged, drop)
                XtX = _get_covmat(X_part, X_part)
                XtY = _get_covmat(X_part, y_part)
                norm_pool_factor = np.sqrt(
                    (n_parts*X_part.shape[0] - 1)/(n_parts*(X_part.shape[0] - 1)))

                if self.XtX_ is None:
                    self.XtX_ = XtX*norm_pool_factor
                else:
                    self.XtX_ += XtX*norm_pool_factor

                if self.XtY_ is None:
                    self.XtY_ = XtY*norm_pool_factor
                else:
                    self.XtY_ += XtY*norm_pool_factor

        else:

            X_part, y_part = self.get_XY(X, y, lagged, drop)

            self.XtX_ = _get_covmat(X_part, X_part)
            self.XtY_ = _get_covmat(X_part, y_part)

        return self

    def fit_direct_cov(self, XXcov=None, XYcov=None, lagged=False, drop=True, overwrite=True, part_length=150., clear_after=True):

        self.XtX_ = XXcov
        self.XtY_ = XYcov

        self.fit_intercept = False
        self.intercept_ = None
        self.coef_ = _ridge_fit_SVD(
            self.XtX_, self.XtY_, self.alpha, from_cov=True)
        self.fitted = True

        if clear_after:
            self.clear_cov()

        return self

    def fit_from_cov(self, X=None, y=None, lagged=False, drop=True, overwrite=True, part_length=150., clear_after=True):
        '''
        Fit model from covariance matrices (handy for v. large data).
        Note: This method is intercept-agnostic. It's recommended to standardize the input data and avoid fitting intercept in the first place.
        Otherwise, the intercept can be estimated as mean values for each channel of y.
        Parameters
        ----------
        X : ndarray (nsamples x nfeats), if None, model will be fitted from accumulated XtX & XtY
            Default: None
        y : ndarray (nsamples x nchans), if None, model will be fitted from accumulated XtX & XtY
            Default: None
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        overwrite : bool
            Default: True
            Whether to reset the accumulated covariance matrices (when X and Y are not None)
        part_length : integer | float
            Default: 150 (seconds) ~ 2.5 minutes. Estimate what will fit in RAM.
            Size of the parts in which the data will be chopped for fitting the model (when X and Y are provided).
        Returns
        -------
        coef_ : ndarray (alphas x nlags x nfeats)
        TODO:
        - Introduce overlap between the segments to prevent losing data (minor, but it should yield exact results)
        '''

        # If X and y are not none, chop them into pieces, compute cov matrices and fit the model (memory efficient)
        if (X is not None) and (y is not None):
            if overwrite:
                self.clear_cov()  # If overwrite, wipe clean

            part_length = int(part_length*self.srate)  # seconds to samples

            assert X.shape[0] == y.shape[0]

            # ToDo -> add padding to the segment to spare some data...
            segments = [(part_length * i) + np.arange(part_length)
                        for i in range(X.shape[0] // part_length)]  # Indices making up segments

            if X.shape[0] % part_length > part_length/2:
                # Create new segment from leftover data (if more than 1/2 part length)
                segments = segments + [np.arange(segments[-1][-1], X.shape[0])]
            else:
                # Add letover data to the last segment
                segments[-1] = np.concatenate((segments[-1],
                                               np.arange(segments[-1][-1], X.shape[0])))

            X = [X[segments[i], :] for i in range(len(segments))]
            y = [y[segments[i], :] for i in range(len(segments))]

            self.add_cov(X, y, lagged=False, drop=True, n_parts=len(segments))

        self.fit_intercept = False
        self.intercept_ = None
        self.coef_ = _ridge_fit_SVD(
            self.XtX_, self.XtY_, self.alpha, from_cov=True)
        self.fitted = True

        if clear_after:
            self.clear_cov()

        return self

    def clear_cov(self):
        '''
        Wipe clean / reset covariance matrices.
        '''
        # print("Clearing saved covariance matrices...")
        self.XtX_ = None
        self.XtY_ = None
        return self

    def predict(self, X):
        """Compute output based on fitted coefficients and feature matrix X.
        Parameters
        ----------
        X : ndarray
            Matrix of features (can be already lagged or not).
        Returns
        -------
        ndarray
            Reconstruction of target with current beta estimates
        Notes
        -----
        If the matrix onky has features in its column (not yet lagged), the lagged version
        of the feature matrix will be created on the fly (this might take some time if the matrix
        is large).
        """
        assert self.fitted, "Fit model first!"

        if self.fit_intercept:
            betas = np.concatenate((self.intercept_, self.coef_), axis=0)
        else:
            #betas = self.get_coef()[:]
            betas = self.coef_[:]

        # Check if input has been lagged already, if not, do it:
        if X.shape[1] != int(self.fit_intercept) + len(self.lags) * self.n_feats_:
            # LOGGER.info("Creating lagged feature matrix...")
            X = lag_matrix(X, lag_samples=self.lags, filling=0.)
            if self.fit_intercept:  # Adding intercept feature:
                X = np.hstack([np.ones((len(X), 1)), X])

        # Do it for every alpha
        pred = np.stack([X.dot(betas[:, :, :, i])
                         for i in range(betas.shape[-1])], axis=-1)

        return pred  # Shape T x Nchan x Alpha

    def score(self, Xtest, ytrue, scoring="corr"):
        """Compute a score of the model given true target and estimated target from Xtest.
        Parameters
        ----------
        Xtest : ndarray
            Array used to get "yhat" estimate from model
        ytrue : ndarray
            True target
        scoring : str (or func in future?)
            Scoring function to be used ("corr", "rmse")
        Returns
        -------
        float
            Score value computed on whole segment.
        """
        yhat = self.predict(Xtest)
        if scoring == 'corr':
            return np.stack([_corr_multifeat(yhat[..., a], ytrue, nchans=self.n_chans_) for a in range(len(self.alpha))], axis=-1)
        elif scoring == 'rmse':
            return np.stack([_rmse_multifeat(yhat[..., a], ytrue) for a in range(len(self.alpha))], axis=-1)
        else:
            raise NotImplementedError(
                "Only correlation score is valid for now...")

    def xval_eval(self, X, y, n_splits=5, lagged=False, drop=True, train_full=True, scoring="corr", segment_length=None, fit_mode='direct', verbose=True):
        '''
        Standard cross-validation. Scoring
        Parameters
        ----------
        X : ndarray (nsamples x nfeats)
        y : ndarray (nsamples x nchans)
        n_splits : integer (default: 5)
            Number of folds
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        train_full : bool (default: True)
            Train model using all the available data after the end of x-val
        scoring : string (default: "corr")
            Scoring method (see scoring())
        segment_length: integer, float (default: None)
            Length of a testing segments (that testing data will be chopped into). If None, use all the available data.
        fit_mode : string {'direct' | 'from_cov_xxx'} (default: 'direct')
            Model training mode. Options:
            'direct' - fit using all the avaiable data at once (i.e. fit())
            'from_cov_xxx' - fit using all the avaiable data from covariance matrices. 
            The routine will chop data into pieces, compute piece-wise cov matrices and fit the model.
            'xxx' portion of the string indicates the lenght of the segments that the data will be chopped into. 
            If not declared (i.e. 'from_cov') the default 2.5 minutes will be used.
        verbose : bool (defaul: True)
        Returns
        -------
        scores - ndarray (n_splits x segments x nchans x alpha)
        ToDo:
        - implement standard scaler / normalizer (optional)
        - handle different scores
        '''

        if np.ndim(self.alpha) < 1 or len(self.alpha) <= 1:
            raise ValueError(
                "Supply several alphas to TRF constructor to use this method.")

        if segment_length:
            segment_length = segment_length*self.srate  # time to samples

        self.fill_lags()

        self.n_feats_ = X.shape[1] if not lagged else X.shape[1] // len(
            self.lags)
        self.n_chans_ = y.shape[1] if y.ndim == 2 else y.shape[2]

        kf = KFold(n_splits=n_splits)
        if segment_length:
            scores = []
        else:
            scores = np.zeros((n_splits, self.n_chans_, len(self.alpha)))

        for kfold, (train, test) in enumerate(kf.split(X)):
            if verbose:
                print("Training/Evaluating fold %d/%d" % (kfold+1, n_splits))

            # Fit using trick with adding covariance matrices -> saves RAM
            if fit_mode.find('from_cov') > -1:
                # Format 'from_cov_xxx' -> xxx - duration of a single part.
                # The main idea is to chunk the data into bite-sized parts that will fit in the RAM
                if len(fit_mode.split('_')) == 2:
                    part_lenght = 150
                elif len(fit_mode.split('_')) == 3:
                    part_lenght = int(fit_mode.split('_')[-1])
                self.fit_from_cov(X[train, :], y[train, :],
                                  overwrite=True, part_length=part_lenght)
            else:  # Fit directly -> slightly faster, but uses more RAM
                self.fit(X[train, :], y[train, :])

            if segment_length:  # Chop testing data into smaller pieces

                if (len(test) % segment_length) > 0:  # Crop if there are some odd samples
                    test_crop = test[:-int(len(test) % segment_length)]
                else:
                    test_crop = test[:]

                # Reshape to # segments x segment duration
                test_segments = test_crop.reshape(
                    int(len(test_crop) / segment_length), -1)

                ccs = [self.score(X[test_segments[i], :], y[test_segments[i], :]) for i in range(
                    test_segments.shape[0])]  # Evaluate each segment

                scores.append(ccs)
            else:  # Evaluate using the entire testing data
                scores[kfold, :] = self.score(X[test, :], y[test, :])

        if segment_length:
            scores = np.asarray(scores)

        if train_full:
            if verbose:
                print("Fitting full model...")
            # Fit using trick with adding covariance matrices -> saves RAM
            if fit_mode.find('from_cov') > -1:
                self.fit_from_cov(X, y, overwrite=True,
                                  part_length=part_lenght)
            else:
                self.fit(X, y)

        return scores

    def __getitem__(self, feats):
        "Extract a sub-part of TRF instance as a new TRF instance (useful for plotting only some features...)"
        # Argument check
        if self.feat_names_ is None:
            if np.ndim(feats) > 0:
                assert isinstance(
                    feats[0], int), "Type not understood, feat_names are ot defined, can only index with int"
                indices = feats

            else:
                assert isinstance(
                    feats, int), "Type not understood, feat_names are ot defined, can only index with int"
                indices = [feats]
                feats = [feats]
        else:
            if np.ndim(feats) > 0:
                assert all([f in self.feat_names_ for f in feats]
                           ), "an element in argument %s in not present in %s" % (feats, self.feat_names_)
                indices = [self.feat_names_.index(f) for f in feats]
            else:
                assert feats in self.feat_names_, "argument %s not present in %s" % (
                    feats, self.feat_names_)
                indices = [self.feat_names_.index(feats)]
                feats = [feats]

        trf = TRFEstimator(tmin=self.tmin, tmax=self.tmax,
                           srate=self.srate, alpha=self.alpha)
        trf.coef_ = self.coef_[:, indices]
        trf.feat_names_ = feats
        trf.n_feats_ = len(feats)
        trf.n_chans_ = self.n_chans_
        trf.fitted = True
        trf.times = self.times
        trf.lags = self.lags
        trf.intercept_ = self.intercept_

        return trf

    def __repr__(self):
        obj = """TRFEstimator(
            alpha=%s,
            fit_intercept=%s,
            srate=%s,
            tmin=%s,
            tmax=%s,
            n_feats=%s,
            n_chans=%s,
            n_lags=%s,
            features : %s
        )
        """ % (self.alpha, self.fit_intercept, self.srate, self.tmin, self.tmax,
               self.n_feats_, self.n_chans_, len(self.lags) if self.lags is not None else None, str(self.feat_names_))
        return obj

    def __add__(self, other_trf):
        "Make available the '+' operator. Will simply add coefficients. Be mindful of dividing by the number of elements later if you want the true mean."
        assert (other_trf.n_feats_ == self.n_feats_ and other_trf.n_chans_ ==
                self.n_chans_), "Both TRF objects must have the same number of features and channels"
        trf = TRFEstimator(tmin=self.tmin, tmax=self.tmax,
                           srate=self.srate, alpha=self.alpha)
        trf.coef_ = np.sum([self.coef_, other_trf.coef_], 0)
        trf.intercept_ = np.sum([self.intercept_, other_trf.intercept_], 0)
        trf.feat_names_ = self.feat_names_
        trf.n_feats_ = self.n_feats_
        trf.n_chans_ = self.n_chans_
        trf.fitted = True
        trf.times = self.times
        trf.lags = self.lags

        return trf

    def plot(self, feat_id=None, alpha_id=None, ax=None, spatial_colors=False, info=None, **kwargs):
        """Plot the TRF of the feature requested as a *butterfly* plot.
        Parameters
        ----------
        feat_id : list or int
            Index of the feature requested or list of features.
            Default is to use all features.
        ax : array of axes (flatten)
            list of subaxes
        **kwargs : **dict
            Parameters to pass to :func:`plt.subplots`
        Returns
        -------
        fig : figure
        """
        if isinstance(feat_id, int):
            # cast into list to be able to use min, len, etc...
            feat_id = list(feat_id)
            if ax is not None:
                fig = ax.figure
        if not feat_id:
            feat_id = range(self.n_feats_)
        if len(feat_id) > 1:
            if ax is not None:
                fig = ax[0].figure
        assert self.fitted, "Fit the model first!"
        assert all([min(feat_id) >= 0, max(feat_id) <
                    self.n_feats_]), "Feat ids not in range"

        if ax is None:
            if 'figsize' not in kwargs.keys():
                fig, ax = plt.subplots(nrows=1, ncols=np.size(feat_id), figsize=(
                    plt.rcParams['figure.figsize'][0] * np.size(feat_id), plt.rcParams['figure.figsize'][1]), **kwargs)
            else:
                fig, ax = plt.subplots(
                    nrows=1, ncols=np.size(feat_id), **kwargs)

        if spatial_colors:
            assert info is not None, "To use spatial colouring, you must supply raw.info instance"
            colors = get_spatial_colors(info)

        for k, feat in enumerate(feat_id):
            if len(feat_id) == 1:
                ax.plot(self.times, self.coef_[:, feat, :])
                if self.feat_names_:
                    ax.set_title('TRF for {:s}'.format(self.feat_names_[feat]))
                if spatial_colors:
                    lines = ax.get_lines()
                    for kc, l in enumerate(lines):
                        l.set_color(colors[kc])
            else:
                ax[k].plot(self.times, self.get_coef()[:, feat, :, 0])
                if self.feat_names_:
                    ax[k].set_title('{:s}'.format(self.feat_names_[feat]))
                if spatial_colors:
                    lines = ax[k].get_lines()
                    for kc, l in enumerate(lines):
                        l.set_color(colors[kc])

        return fig


class DecoderEstimator(BaseEstimator):

    def __init__(self, times=(0.,), tmin=None, tmax=None, srate=1., alpha=[0.], fit_intercept=False, mtype='backward'):

        # Times reflect mismatch a -> b, where a - dependent, b - predicted
        # Negative timelags indicate a lagging behind b
        # Positive timelags indicate b lagging behind a
        # For example:
        # eeg -> env (tmin = -0.5, tmax = 0.1)
        # Indicate timeframe from -100 ms (eeg precedes stimulus): 500 ms (eeg after stimulus)
        self.tmin = tmin
        self.tmax = tmax
        self.times = times
        self.srate = srate
        self.alpha = alpha
        # Forward or backward. Required for formatting coefficients in get_coef (convention: forward - stimulus -> eeg, backward - eeg - stimulus)
        self.mtype = mtype
        self.fit_intercept = fit_intercept
        self.fitted = False
        self.lags = None

        # All following attributes are only defined once fitted (hence the "_" suffix)
        self.intercept_ = None
        self.coef_ = None
        self.n_feats_ = None
        self.n_chans_ = None
        self.feat_names_ = None
        self.valid_samples_ = None
        # Autocorrelation matrix of feature X (thus XtX) -> used for computing model using fit_from_cov
        self.XtX_ = None
        # Covariance matrix of features X and Y (thus XtX) -> used for computing model using fit_from_cov
        self.XtY_ = None

    def fill_lags(self):
        """Fill the lags attributes, with number of samples and times in seconds.
        Note
        ----
        Necessary to call this function if one wishes to use trf.lags _before_
        :func:`trf.fit` is called.

        """
        if self.tmin and self.tmax:
            # LOGGER.info("Will use lags spanning form tmin to tmax.\nTo use individual lags, use the `times` argument...")
            self.lags = lag_span(self.tmin, self.tmax, srate=self.srate)[
                ::-1]  # pylint: disable=invalid-unary-operand-type
            # self.lags = lag_span(-tmax, -tmin, srate=srate) #pylint: disable=invalid-unary-operand-type
            self.times = self.lags[::-1] / self.srate
        else:
            self.times = np.asarray(self.times)
            self.lags = lag_sparse(self.times, self.srate)[::-1]

    def get_XY(self, X, y, lagged=False, drop=True, feat_names=()):
        '''
        Preprocess X and y before fitting (finding mapping between X -> y)
        Parameters
        ----------
        X : ndarray (T x nfeat)
        y : ndarray (T x nchan)
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        feat_names : list
            Names of features being fitted. Must be of length ``nfeats``.
        Returns
        -------
        Features preprocessed for fitting the model.
        X : ndarray (T x nlags * nfeats)
        y : ndarray (T x nchan)
        '''
        self.fill_lags()

        X = np.asarray(X)
        y = np.asarray(y)

        # Estimate the necessary size to compute stuff
        y_memory = sum([yy.nbytes for yy in y]) if np.ndim(
            y) == 3 else y.nbytes
        estimated_mem_usage = X.nbytes * \
            (len(self.lags) if not lagged else 1) + y_memory
        if estimated_mem_usage/1024.**3 > mem_check():
            raise MemoryError("Not enough RAM available! (needed %.1fGB, but only %.1fGB available)" % (
                estimated_mem_usage/1024.**3, mem_check()))

        # Fill n_feat and n_chan attributes
        # if X has been lagged, divide by number of lags
        self.n_feats_ = X.shape[1] if not lagged else X.shape[1] // len(
            self.lags)
        self.n_chans_ = y.shape[1] if y.ndim == 2 else y.shape[2]

        # Assess if feat names corresponds to feat number
        if feat_names:
            err_msg = "Length of feature names does not match number of columns from feature matrix"
            if lagged:
                assert len(feat_names) == X.shape[1] // len(self.lags), err_msg
            else:
                assert len(feat_names) == X.shape[1], err_msg
            self.feat_names_ = feat_names

        # this include non-valid samples for now
        n_samples_all = y.shape[0] if y.ndim == 2 else y.shape[1]

        # drop samples that can't be reconstructed because on the edge, all is true otherwise
        if drop:
            self.valid_samples_ = np.logical_not(np.logical_or(np.arange(n_samples_all) < abs(max(self.lags)),
                                                               np.arange(n_samples_all)[::-1] < abs(min(self.lags))))
        else:
            self.valid_samples_ = np.ones((n_samples_all,), dtype=bool)

        # Creating lag-matrix droping NaN values if necessary
        y = y[self.valid_samples_, :] if y.ndim == 2 else y[:,
                                                            self.valid_samples_, :]
        if not lagged:
            X = lag_matrix(X, lag_samples=self.lags,
                           drop_missing=drop, filling=np.nan if drop else 0.)

        return X, y

    def fit(self, X, y, lagged=False, drop=True, feat_names=()):
        """Fit the TRF model.
        Mapping X -> y. Note the convention of timelags and type of model for seamless recovery of coefficients.
        Parameters
        ----------
        X : ndarray (nsamples x nfeats)
        y : ndarray (nsamples x nchans)
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        feat_names : list
            Names of features being fitted. Must be of length ``nfeats``.
        Returns
        -------
        coef_ : ndarray (alphas x nlags x nfeats)
        intercept_ : ndarray (nfeats x 1)
        """

        # Preprocess and lag inputs
        X, y = self.get_XY(X, y, lagged, drop, feat_names)

        # Adding intercept feature:
        if self.fit_intercept:
            X = np.hstack([np.ones((len(X), 1)), X])

        # Regress with Ridge to obtain coef for the input alpha
        self.coef_ = _ridge_fit_SVD(X, y, self.alpha)

        # Reshaping and getting coefficients
        if self.fit_intercept:
            self.intercept_ = self.coef_[0, np.newaxis, :]
            self.coef_ = self.coef_[1:, :]

        self.fitted = True

        return self

    def get_coef(self):
        '''
        Format and return coefficients. Note mtype attribute needs to be declared in the __init__.

        Returns
        -------
        coef_ : ndarray (nlags x nfeats x nchans x regularization params)
        '''
        if np.ndim(self.alpha) == 0:
            betas = np.reshape(self.coef_, (len(self.lags),
                                            self.n_feats_, self.n_chans_))
        else:
            betas = np.reshape(self.coef_, (len(self.lags),
                                            self.n_feats_, self.n_chans_, len(self.alpha)))

        if self.mtype == 'forward':
            betas = betas[::-1, :]

        return betas

    def add_cov(self, X, y, lagged=False, drop=True, n_parts=1):
        '''
        Compute and add (with normalization factor) covariance matrices XtX, XtY
        For v. large population models when it's not possible to load all the data to memory at once.
        Parameters
        ----------
        X : ndarray (nsamples x nfeats) or list/tuple of ndarray (from which the model will be computed)
        y : ndarray (nsamples x nchans) or list/tuple of ndarray (from which the model will be computed)
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        n_parts : number of parts from which the covariance matrix are computed (required for normalization)
            Default: 1
        Returns
        -------
        XtX : autocorrelation matrix for X (accumulated)
        XtY : covariance matrix for X & Y (accumulated)
        '''
        if isinstance(X, (list, tuple)) and n_parts > 1:

            assert len(X) == len(y)

            for part in range(len(X)):
                assert len(X[part]) == len(y[part])
                X_part, y_part = self.get_XY(X[part], y[part], lagged, drop)
                XtX = _get_covmat(X_part, X_part)
                XtY = _get_covmat(X_part, y_part)
                norm_pool_factor = np.sqrt(
                    (n_parts*X_part.shape[0] - 1)/(n_parts*(X_part.shape[0] - 1)))

                if self.XtX_ is None:
                    self.XtX_ = XtX*norm_pool_factor
                else:
                    self.XtX_ += XtX*norm_pool_factor

                if self.XtY_ is None:
                    self.XtY_ = XtY*norm_pool_factor
                else:
                    self.XtY_ += XtY*norm_pool_factor

        else:

            X_part, y_part = self.get_XY(X, y, lagged, drop)

            self.XtX_ = _get_covmat(X_part, X_part)
            self.XtY_ = _get_covmat(X_part, y_part)

        return self

    def fit_direct_cov(self, XXcov=None, XYcov=None, lagged=False, drop=True, overwrite=True, part_length=150., clear_after=True):

        self.XtX_ = XXcov
        self.XtY_ = XYcov

        self.fit_intercept = False
        self.intercept_ = None
        self.coef_ = _ridge_fit_SVD(
            self.XtX_, self.XtY_, self.alpha, from_cov=True)
        self.fitted = True

        if clear_after:
            self.clear_cov()

        return self

    def fit_from_cov(self, X=None, y=None, lagged=False, drop=True, overwrite=True, part_length=150., clear_after=True):
        '''
        Fit model from covariance matrices (handy for v. large data).
        Note: This method is intercept-agnostic. It's recommended to standardize the input data and avoid fitting intercept in the first place.
        Otherwise, the intercept can be estimated as mean values for each channel of y.
        Parameters
        ----------
        X : ndarray (nsamples x nfeats), if None, model will be fitted from accumulated XtX & XtY
            Default: None
        y : ndarray (nsamples x nchans), if None, model will be fitted from accumulated XtX & XtY
            Default: None
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        overwrite : bool
            Default: True
            Whether to reset the accumulated covariance matrices (when X and Y are not None)
        part_length : integer | float
            Default: 150 (seconds) ~ 2.5 minutes. Estimate what will fit in RAM.
            Size of the parts in which the data will be chopped for fitting the model (when X and Y are provided).
        Returns
        -------
        coef_ : ndarray (alphas x nlags x nfeats)
        TODO:
        - Introduce overlap between the segments to prevent losing data (minor, but it should yield exact results)
        '''

        # If X and y are not none, chop them into pieces, compute cov matrices and fit the model (memory efficient)
        if (X is not None) and (y is not None):
            if overwrite:
                self.clear_cov()  # If overwrite, wipe clean

            part_length = int(part_length*self.srate)  # seconds to samples

            assert X.shape[0] == y.shape[0]

            # ToDo -> add padding to the segment to spare some data...
            segments = [(part_length * i) + np.arange(part_length)
                        for i in range(X.shape[0] // part_length)]  # Indices making up segments

            if X.shape[0] % part_length > part_length/2:
                # Create new segment from leftover data (if more than 1/2 part length)
                segments = segments + [np.arange(segments[-1][-1], X.shape[0])]
            else:
                # Add letover data to the last segment
                segments[-1] = np.concatenate((segments[-1],
                                               np.arange(segments[-1][-1], X.shape[0])))

            X = [X[segments[i], :] for i in range(len(segments))]
            y = [y[segments[i], :] for i in range(len(segments))]

            self.add_cov(X, y, lagged=False, drop=True, n_parts=len(segments))

        self.fit_intercept = False
        self.intercept_ = None
        self.coef_ = _ridge_fit_SVD(
            self.XtX_, self.XtY_, self.alpha, from_cov=True)
        self.fitted = True

        if clear_after:
            self.clear_cov()

        return self

    def clear_cov(self):
        '''
        Wipe clean / reset covariance matrices.
        '''
        # print("Clearing saved covariance matrices...")
        self.XtX_ = None
        self.XtY_ = None
        return self

    def predict(self, X):
        """Compute output based on fitted coefficients and feature matrix X.
        Parameters
        ----------
        X : ndarray
            Matrix of features (can be already lagged or not).
        Returns
        -------
        ndarray
            Reconstruction of target with current beta estimates
        Notes
        -----
        If the matrix onky has features in its column (not yet lagged), the lagged version
        of the feature matrix will be created on the fly (this might take some time if the matrix
        is large).
        """
        assert self.fitted, "Fit model first!"

        if self.fit_intercept:
            betas = np.concatenate((self.intercept_, self.coef_), axis=0)
        else:
            #betas = self.get_coef()[:]
            betas = self.coef_[:]

        # Check if input has been lagged already, if not, do it:
        if X.shape[1] != int(self.fit_intercept) + len(self.lags) * self.n_feats_:
            # LOGGER.info("Creating lagged feature matrix...")
            X = lag_matrix(X, lag_samples=self.lags, filling=0.)
            if self.fit_intercept:  # Adding intercept feature:
                X = np.hstack([np.ones((len(X), 1)), X])

        # Do it for every alpha
        pred = np.stack([X.dot(betas[:, :, :, i])
                         for i in range(betas.shape[-1])], axis=-1)

        return pred  # Shape T x Nchan x Alpha

    def score(self, Xtest, ytrue, scoring="corr"):
        """Compute a score of the model given true target and estimated target from Xtest.
        Parameters
        ----------
        Xtest : ndarray
            Array used to get "yhat" estimate from model
        ytrue : ndarray
            True target
        scoring : str (or func in future?)
            Scoring function to be used ("corr", "rmse")
        Returns
        -------
        float
            Score value computed on whole segment.
        """
        yhat = self.predict(Xtest)
        if scoring == 'corr':
            return np.stack([_corr_multifeat(yhat[..., a], ytrue, nchans=self.n_chans_) for a in range(len(self.alpha))], axis=-1)
        elif scoring == 'rmse':
            return np.stack([_rmse_multifeat(yhat[..., a], ytrue) for a in range(len(self.alpha))], axis=-1)
        else:
            raise NotImplementedError(
                "Only correlation score is valid for now...")

    def xval_eval(self, X, y, n_splits=5, lagged=False, drop=True, train_full=True, scoring="corr", segment_length=None, fit_mode='direct', verbose=True):
        '''
        Standard cross-validation. Scoring
        Parameters
        ----------
        X : ndarray (nsamples x nfeats)
        y : ndarray (nsamples x nchans)
        n_splits : integer (default: 5)
            Number of folds
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        train_full : bool (default: True)
            Train model using all the available data after the end of x-val
        scoring : string (default: "corr")
            Scoring method (see scoring())
        segment_length: integer, float (default: None)
            Length of a testing segments (that testing data will be chopped into). If None, use all the available data.
        fit_mode : string {'direct' | 'from_cov_xxx'} (default: 'direct')
            Model training mode. Options:
            'direct' - fit using all the avaiable data at once (i.e. fit())
            'from_cov_xxx' - fit using all the avaiable data from covariance matrices. 
            The routine will chop data into pieces, compute piece-wise cov matrices and fit the model.
            'xxx' portion of the string indicates the lenght of the segments that the data will be chopped into. 
            If not declared (i.e. 'from_cov') the default 2.5 minutes will be used.
        verbose : bool (defaul: True)
        Returns
        -------
        scores - ndarray (n_splits x segments x nchans x alpha)
        ToDo:
        - implement standard scaler / normalizer (optional)
        - handle different scores
        '''

        if np.ndim(self.alpha) < 1 or len(self.alpha) <= 1:
            raise ValueError(
                "Supply several alphas to TRF constructor to use this method.")

        if segment_length:
            segment_length = segment_length*self.srate  # time to samples

        self.fill_lags()

        self.n_feats_ = X.shape[1] if not lagged else X.shape[1] // len(
            self.lags)
        self.n_chans_ = y.shape[1] if y.ndim == 2 else y.shape[2]

        kf = KFold(n_splits=n_splits)
        if segment_length:
            scores = []
        else:
            scores = np.zeros((n_splits, self.n_chans_, len(self.alpha)))

        for kfold, (train, test) in enumerate(kf.split(X)):
            if verbose:
                print("Training/Evaluating fold %d/%d" % (kfold+1, n_splits))

            # Fit using trick with adding covariance matrices -> saves RAM
            if fit_mode.find('from_cov') > -1:
                # Format 'from_cov_xxx' -> xxx - duration of a single part.
                # The main idea is to chunk the data into bite-sized parts that will fit in the RAM
                if len(fit_mode.split('_')) == 2:
                    part_lenght = 150
                elif len(fit_mode.split('_')) == 3:
                    part_lenght = int(fit_mode.split('_')[-1])
                self.fit_from_cov(X[train, :], y[train, :],
                                  overwrite=True, part_length=part_lenght)
            else:  # Fit directly -> slightly faster, but uses more RAM
                self.fit(X[train, :], y[train, :])

            if segment_length:  # Chop testing data into smaller pieces

                if (len(test) % segment_length) > 0:  # Crop if there are some odd samples
                    test_crop = test[:-int(len(test) % segment_length)]
                else:
                    test_crop = test[:]

                # Reshape to # segments x segment duration
                test_segments = test_crop.reshape(
                    int(len(test_crop) / segment_length), -1)

                ccs = [self.score(X[test_segments[i], :], y[test_segments[i], :]) for i in range(
                    test_segments.shape[0])]  # Evaluate each segment

                scores.append(ccs)
            else:  # Evaluate using the entire testing data
                scores[kfold, :] = self.score(X[test, :], y[test, :])

        if segment_length:
            scores = np.asarray(scores)

        if train_full:
            if verbose:
                print("Fitting full model...")
            # Fit using trick with adding covariance matrices -> saves RAM
            if fit_mode.find('from_cov') > -1:
                self.fit_from_cov(X, y, overwrite=True,
                                  part_length=part_lenght)
            else:
                self.fit(X, y)

        return scores

    def __getitem__(self, feats):
        "Extract a sub-part of TRF instance as a new TRF instance (useful for plotting only some features...)"
        # Argument check
        if self.feat_names_ is None:
            if np.ndim(feats) > 0:
                assert isinstance(
                    feats[0], int), "Type not understood, feat_names are ot defined, can only index with int"
                indices = feats

            else:
                assert isinstance(
                    feats, int), "Type not understood, feat_names are ot defined, can only index with int"
                indices = [feats]
                feats = [feats]
        else:
            if np.ndim(feats) > 0:
                assert all([f in self.feat_names_ for f in feats]
                           ), "an element in argument %s in not present in %s" % (feats, self.feat_names_)
                indices = [self.feat_names_.index(f) for f in feats]
            else:
                assert feats in self.feat_names_, "argument %s not present in %s" % (
                    feats, self.feat_names_)
                indices = [self.feat_names_.index(feats)]
                feats = [feats]

        trf = TRFEstimator(tmin=self.tmin, tmax=self.tmax,
                           srate=self.srate, alpha=self.alpha)
        trf.coef_ = self.coef_[:, indices]
        trf.feat_names_ = feats
        trf.n_feats_ = len(feats)
        trf.n_chans_ = self.n_chans_
        trf.fitted = True
        trf.times = self.times
        trf.lags = self.lags
        trf.intercept_ = self.intercept_

        return trf

    def __repr__(self):
        obj = """TRFEstimator(
            alpha=%s,
            fit_intercept=%s,
            srate=%s,
            tmin=%s,
            tmax=%s,
            n_feats=%s,
            n_chans=%s,
            n_lags=%s,
            features : %s
        )
        """ % (self.alpha, self.fit_intercept, self.srate, self.tmin, self.tmax,
               self.n_feats_, self.n_chans_, len(self.lags) if self.lags is not None else None, str(self.feat_names_))
        return obj

    def __add__(self, other_trf):
        "Make available the '+' operator. Will simply add coefficients. Be mindful of dividing by the number of elements later if you want the true mean."
        assert (other_trf.n_feats_ == self.n_feats_ and other_trf.n_chans_ ==
                self.n_chans_), "Both TRF objects must have the same number of features and channels"
        trf = TRFEstimator(tmin=self.tmin, tmax=self.tmax,
                           srate=self.srate, alpha=self.alpha)
        trf.coef_ = np.sum([self.coef_, other_trf.coef_], 0)
        trf.intercept_ = np.sum([self.intercept_, other_trf.intercept_], 0)
        trf.feat_names_ = self.feat_names_
        trf.n_feats_ = self.n_feats_
        trf.n_chans_ = self.n_chans_
        trf.fitted = True
        trf.times = self.times
        trf.lags = self.lags

        return trf


class Decoder(BaseEstimator):
    """Temporal Response Function (TRF) Estimator Class.
    This class allows to estimate TRF from a set of feature signals and an EEG dataset in the same fashion
    than ReceptiveFieldEstimator does in MNE.
    However, an arbitrary set of lags can be given. Namely, it can be used in two ways:
    - calling with `tmin` and tmax` arguments will compute lags spanning from `tmin` to `tmax`
    - with the `times` argument, one can request an arbitrary set of time lags at which to compute
    the coefficients of the TRF
    Attributes
    ----------
    lags : 1d-array
        Array of `int`, corresponding to lag in samples at which the TRF coefficients are computed
    times : 1d-array
        Array of `float`, corresponding to lag in seconds at which the TRF coefficients are computed
    srate : float
        Sampling rate
    fit_intercept : bool
        Whether a column of ones should be added to the design matrix to fit an intercept
    fitted : bool
        True once the TRF has been fitted on EEG data
    intercept_ : 1d array (nchans, )
        Intercepts
    coef_ : ndarray (nlags, nfeats, nchans)
        Actual TRF coefficients
    n_feats_ : int
        Number of word level features in TRF
    n_chans_: int
        Number of EEG channels in TRF
    feat_names_ : list
        Names of each word level features
    Notes
    -----
        - Attributes with a `_` suffix are only set once the TRF has been fitted on EEG data (i.e. after
        the method :meth:`TRFEstimator.fit` has been called).

        - Can fit on a list of multiple dataset, where we have a list of target Y and
        a single stimulus matrix of features X, then the computation is made such that
        the coefficients computed are similar to those obtained by concatenating all matrices

        - Times reflect mismatch a -> b, where a - dependent, b - predicted
        Negative timelags indicate a lagging behind b
        Positive timelags indicate b lagging behind a
        For example:
        eeg -> env (tmin = -0.5, tmax = 0.1)
        Indicate timeframe from -100 ms (eeg precedes stimulus): 500 ms (eeg after stimulus)
    Examples
    --------
    >>> trf = TRFEstimator(tmin=-0.5, tmax-1.2, srate=125)
    >>> x = np.random.randn(1000, 3)
    >>> y = np.random.randn(1000, 2)
    >>> trf.fit(x, y, lagged=False)
    """

    def __init__(self, times=(0.,), tmin=None, tmax=None, srate=1., alpha=[0.], fit_intercept=False, mtype='forward'):

        # Times reflect mismatch a -> b, where a - dependent, b - predicted
        # Negative timelags indicate a lagging behind b
        # Positive timelags indicate b lagging behind a
        # For example:
        # eeg -> env (tmin = -0.5, tmax = 0.1)
        # Indicate timeframe from -100 ms (eeg precedes stimulus): 500 ms (eeg after stimulus)
        self.tmin = tmin
        self.tmax = tmax
        self.times = times
        self.srate = srate
        self.alpha = alpha
        # Forward or backward. Required for formatting coefficients in get_coef (convention: forward - stimulus -> eeg, backward - eeg - stimulus)
        self.mtype = mtype
        self.fit_intercept = fit_intercept
        self.fitted = False
        self.lags = None

        # All following attributes are only defined once fitted (hence the "_" suffix)
        self.intercept_ = None
        self.coef_ = None
        self.n_feats_ = None
        self.n_chans_ = None
        self.feat_names_ = None
        self.valid_samples_ = None
        # Autocorrelation matrix of feature X (thus XtX) -> used for computing model using fit_from_cov
        self.XtX_ = None
        # Covariance matrix of features X and Y (thus XtX) -> used for computing model using fit_from_cov
        self.XtY_ = None
        self.coef_additive = None

    def fill_lags(self):
        """Fill the lags attributes.
        Note
        ----
        Necessary to call this function if one wishes to use trf.lags _before_
        :func:`trf.fit` is called.

        """
        if self.tmin and self.tmax:
            # LOGGER.info("Will use lags spanning form tmin to tmax.\nTo use individual lags, use the `times` argument...")
            self.lags = lag_span(self.tmin, self.tmax, srate=self.srate)[
                ::-1]  # pylint: disable=invalid-unary-operand-type
            # self.lags = lag_span(-tmax, -tmin, srate=srate) #pylint: disable=invalid-unary-operand-type
            self.times = self.lags[::-1] / self.srate
        else:
            self.times = np.asarray(self.times)
            self.lags = lag_sparse(self.times, self.srate)[::-1]

    def get_XY(self, X, y, lagged=False, drop=True, feat_names=()):
        '''
        Preprocess X and y before fitting (finding mapping between X -> y)
        Parameters
        ----------
        X : ndarray (T x nfeat)
        y : ndarray (T x nchan)
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        feat_names : list
            Names of features being fitted. Must be of length ``nfeats``.
        Returns
        -------
        Features preprocessed for fitting the model.
        X : ndarray (T x nlags * nfeats)
        y : ndarray (T x nchan)
        '''
        self.fill_lags()

        X = np.asarray(X)
        y = np.asarray(y)

        y_memory = sum([yy.nbytes for yy in y]) if np.ndim(
            y) == 3 else y.nbytes
        estimated_mem_usage = X.nbytes * \
            (len(self.lags) if not lagged else 1) + y_memory
        if estimated_mem_usage/1024.**3 > mem_check():
            raise MemoryError("Not enough RAM available! (needed %.1fGB, but only %.1fGB available)" % (
                estimated_mem_usage/1024.**3, mem_check()))

        self.n_feats_ = X.shape[1] if not lagged else X.shape[1] // len(
            self.lags)
        self.n_chans_ = y.shape[1] if y.ndim == 2 else y.shape[2]

        if feat_names:
            err_msg = "Length of feature names does not match number of columns from feature matrix"
            if lagged:
                assert len(feat_names) == X.shape[1] // len(self.lags), err_msg
            else:
                assert len(feat_names) == X.shape[1], err_msg
            self.feat_names_ = feat_names

        # this include non-valid samples for now
        n_samples_all = y.shape[0] if y.ndim == 2 else y.shape[1]

        if drop:
            self.valid_samples_ = np.logical_not(np.logical_or(np.arange(n_samples_all) < abs(max(self.lags)),
                                                               np.arange(n_samples_all)[::-1] < abs(min(self.lags))))
        else:
            self.valid_samples_ = np.ones((n_samples_all,), dtype=bool)

        # Creating lag-matrix droping NaN values if necessary
        y = y[self.valid_samples_, :] if y.ndim == 2 else y[:,
                                                            self.valid_samples_, :]
        if not lagged:
            X = lag_matrix(X, lag_samples=self.lags,
                           drop_missing=drop, filling=np.nan if drop else 0.)

        return X, y

    def fit_direct_cov(self, XXcov=None, XYcov=None, lagged=False, drop=True, overwrite=True, part_length=150., clear_after=True):

        self.XtX_ = XXcov
        self.XtY_ = XYcov

        self.fit_intercept = False
        self.intercept_ = None
        self.coef_ = _ridge_fit_SVD(
            self.XtX_, self.XtY_, self.alpha, from_cov=True)
        self.fitted = True

        if clear_after:
            self.clear_cov()

        return self

    def fit(self, X, y, lagged=False, drop=True, feat_names=()):
        """Fit the TRF model.
        Mapping X -> y. Note the convention of timelags and type of model for seamless recovery of coefficients.
        Parameters
        ----------
        X : ndarray (nsamples x nfeats)
        y : ndarray (nsamples x nchans)
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        feat_names : list
            Names of features being fitted. Must be of length ``nfeats``.
        Returns
        -------
        coef_ : ndarray (alphas x nlags x nfeats)
        intercept_ : ndarray (nfeats x 1)
        """

        # Preprocess and lag inputs
        X, y = self.get_XY(X, y, lagged, drop, feat_names)

        # Adding intercept feature:
        if self.fit_intercept:
            X = np.hstack([np.ones((len(X), 1)), X])

        self.coef_ = _ridge_fit_SVD(X, y, self.alpha)

        # Reshaping and getting coefficients
        if self.fit_intercept:
            self.intercept_ = self.coef_[0, np.newaxis, :]
            self.coef_ = self.coef_[1:, :]

        self.fitted = True

        return self

    def get_coef(self):
        '''
        Format and return coefficients. Note mtype attribute needs to be declared in the __init__.

        Returns
        -------
        coef_ : ndarray (nlags x nfeats x nchans x regularization params)
        '''
        '''
        if np.ndim(self.alpha) == 0:
            betas = np.reshape(self.coef_, (len(self.lags), self.n_feats_, self.n_chans_))
        else:
            betas = np.reshape(self.coef_, (len(self.lags), self.n_feats_, self.n_chans_, len(self.alpha)))

        if self.mtype == 'forward':
            betas = betas[::-1,:]
        '''
        if np.ndim(self.alpha) == 0:
            betas = np.reshape(self.coef_, (len(self.lags),
                                            self.n_feats_, self.n_chans_))
        else:
            betas = np.reshape(self.coef_, (len(self.lags),
                                            self.n_feats_, self.n_chans_, len(self.alpha)))

        if self.mtype == 'forward':
            betas = betas[::-1, :]

        return betas

    def add_cov(self, X, y, lagged=False, drop=True, n_parts=1):
        '''
        Compute and add (with normalization factor) covariance matrices XtX, XtY
        For v. large population models when it's not possible to load all the data to memory at once.
        Parameters
        ----------
        X : ndarray (nsamples x nfeats) or list/tuple of ndarray (from which the model will be computed)
        y : ndarray (nsamples x nchans) or list/tuple of ndarray (from which the model will be computed)
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        n_parts : number of parts from which the covariance matrix are computed (required for normalization)
            Default: 1
        Returns
        -------
        XtX : autocorrelation matrix for X (accumulated)
        XtY : covariance matrix for X & Y (accumulated)
        '''
        if isinstance(X, (list, tuple)) and n_parts > 1:
            assert len(X) == len(y)

            for part in range(len(X)):
                assert len(X[part]) == len(y[part])
                X_part, y_part = self.get_XY(X[part], y[part], lagged, drop)
                XtX = _get_covmat(X_part, X_part)
                XtY = _get_covmat(X_part, y_part)
                norm_pool_factor = np.sqrt(
                    (n_parts*X_part.shape[0] - 1)/(n_parts*(X_part.shape[0] - 1)))

                if self.XtX_ is None:
                    self.XtX_ = XtX*norm_pool_factor
                else:
                    self.XtX_ += XtX*norm_pool_factor

                if self.XtY_ is None:
                    self.XtY_ = XtY*norm_pool_factor
                else:
                    self.XtY_ += XtY*norm_pool_factor

        else:
            X_part, y_part = self.get_XY(X, y, lagged, drop)
            self.XtX = _get_covmat(X_part, X_part)
            self.XtY = _get_covmat(X_part, y_part)

        return self

    def fit_from_cov(self, X=None, y=None, lagged=False, drop=True, overwrite=True, part_length=150., clear_after=True):
        '''
        Fit model from covariance matrices (handy for v. large data).
        Note: This method is intercept-agnostic. It's recommended to standardize the input data and avoid fitting intercept in the first place.
        Otherwise, the intercept can be estimated as mean values for each channel of y.
        Parameters
        ----------
        X : ndarray (nsamples x nfeats), if None, model will be fitted from accumulated XtX & XtY
            Default: None
        y : ndarray (nsamples x nchans), if None, model will be fitted from accumulated XtX & XtY
            Default: None
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        overwrite : bool
            Default: True
            Whether to reset the accumulated covariance matrices (when X and Y are not None)
        part_length : integer | float
            Default: 150 (seconds) ~ 2.5 minutes. Estimate what will fit in RAM.
            Size of the parts in which the data will be chopped for fitting the model (when X and Y are provided).
        Returns
        -------
        coef_ : ndarray (alphas x nlags x nfeats)
        TODO:
        - Introduce overlap between the segments to prevent losing data (minor, but it should yield exact results)
        '''

        # If X and y are not none, chop them into pieces, compute cov matrices and fit the model (memory efficient)
        if (X is not None) and (y is not None):
            if overwrite:
                self.clear_cov()  # If overwrite, wipe clean

            part_length = int(part_length*self.srate)  # seconds to samples

            assert X.shape[0] == y.shape[0]

            # ToDo -> add padding to the segment to spare some data...
            segments = [(part_length * i) + np.arange(part_length)
                        for i in range(X.shape[0] // part_length)]  # Indices making up segments

            if X.shape[0] % part_length > part_length/2:
                # Create new segment from leftover data (if more than 1/2 part length)
                segments = segments + [np.arange(segments[-1][-1], X.shape[0])]
            else:
                # Add letover data to the last segment
                segments[-1] = np.concatenate((segments[-1],
                                               np.arange(segments[-1][-1], X.shape[0])))

            X = [X[segments[i], :] for i in range(len(segments))]
            y = [y[segments[i], :] for i in range(len(segments))]

            self.add_cov(X, y, lagged=False, drop=True, n_parts=len(segments))

        self.fit_intercept = False
        self.intercept_ = None
        self.coef_ = _ridge_fit_SVD(
            self.XtX_, self.XtY_, self.alpha, from_cov=True)
        self.fitted = True

        if clear_after:
            self.clear_cov()

        return self

    def clear_cov(self):
        '''
        Wipe clean / reset covariance matrices.
        '''
        # print("Clearing saved covariance matrices...")
        self.XtX_ = None
        self.XtY_ = None
        return self

    def predict(self, X):
        """Compute output based on fitted coefficients and feature matrix X.
        Parameters
        ----------
        X : ndarray
            Matrix of features (can be already lagged or not).
        Returns
        -------
        ndarray
            Reconstruction of target with current beta estimates
        Notes
        -----
        If the matrix onky has features in its column (not yet lagged), the lagged version
        of the feature matrix will be created on the fly (this might take some time if the matrix
        is large).
        """
        assert self.fitted, "Fit model first!"

        if self.fit_intercept:
            betas = np.concatenate((self.intercept_, self.coef_), axis=0)
        else:
            betas = self.coef_[:]

        # Check if input has been lagged already, if not, do it:
        if X.shape[1] != int(self.fit_intercept) + len(self.lags) * self.n_feats_:
            # LOGGER.info("Creating lagged feature matrix...")
            X = lag_matrix(X, lag_samples=self.lags, filling=0.)
            if self.fit_intercept:  # Adding intercept feature:
                X = np.hstack([np.ones((len(X), 1)), X])

        # Do it for every alpha
        pred = np.stack([X.dot(betas[:, :, i])
                         for i in range(betas.shape[-1])], axis=-1)

        return pred  # Shape T x Nchan x Alpha

    def score(self, Xtest, ytrue, scoring="corr"):
        """Compute a score of the model given true target and estimated target from Xtest.
        Parameters
        ----------
        Xtest : ndarray
            Array used to get "yhat" estimate from model
        ytrue : ndarray
            True target
        scoring : str (or func in future?)
            Scoring function to be used ("corr", "rmse")
        Returns
        -------
        float
            Score value computed on whole segment.
        """
        yhat = self.predict(Xtest)
        if scoring == 'corr':
            return np.stack([_corr_multifeat(yhat[..., a], ytrue, nchans=self.n_chans_) for a in range(len(self.alpha))], axis=-1)
        elif scoring == 'rmse':
            return np.stack([_rmse_multifeat(yhat[..., a], ytrue) for a in range(len(self.alpha))], axis=-1)
        else:
            raise NotImplementedError(
                "Only correlation score is valid for now...")

    def xval_eval(self, X, y, n_splits=5, lagged=False, drop=True, train_full=True, scoring="corr", segment_length=None, fit_mode='direct', verbose=True, Additive_model=False):
        '''
        Standard cross-validation. Scoring
        Parameters
        ----------
        X : ndarray (nsamples x nfeats)
        y : ndarray (nsamples x nchans)
        n_splits : integer (default: 5)
            Number of folds
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        train_full : bool (default: True)
            Train model using all the available data after the end of x-val
        scoring : string (default: "corr")
            Scoring method (see scoring())
        segment_length: integer, float (default: None)
            Length of a testing segments (that testing data will be chopped into). If None, use all the available data.
        fit_mode : string {'direct' | 'from_cov_xxx'} (default: 'direct')
            Model training mode. Options:
            'direct' - fit using all the avaiable data at once (i.e. fit())
            'from_cov_xxx' - fit using all the avaiable data from covariance matrices. 
            The routine will chop data into pieces, compute piece-wise cov matrices and fit the model.
            'xxx' portion of the string indicates the lenght of the segments that the data will be chopped into. 
            If not declared (i.e. 'from_cov') the default 2.5 minutes will be used.
        verbose : bool (defaul: True)
        Returns
        -------
        scores - ndarray (n_splits x segments x nchans x alpha)
        ToDo:
        - implement standard scaler / normalizer (optional)
        - handle different scores
        '''

        if np.ndim(self.alpha) < 1 or len(self.alpha) <= 1:
            raise ValueError(
                "Supply several alphas to TRF constructor to use this method.")

        if segment_length:
            segment_length = segment_length*self.srate  # time to samples

        self.fill_lags()

        self.n_feats_ = X.shape[1] if not lagged else X.shape[1] // len(
            self.lags)
        self.n_chans_ = y.shape[1] if y.ndim == 2 else y.shape[2]

        kf = KFold(n_splits=n_splits)
        if segment_length:
            scores = []
        else:
            scores = np.zeros((n_splits, self.n_chans_, len(self.alpha)))

        for kfold, (train, test) in enumerate(kf.split(X)):

            if not Additive_model:
                if verbose:
                    print("Training/Evaluating fold %d/%d" %
                          (kfold+1, n_splits))

                # Fit using trick with adding covariance matrices -> saves RAM
                if fit_mode.find('from_cov') > -1:
                    # Format 'from_cov_xxx' -> xxx - duration of a single part.
                    # The main idea is to chunk the data into bite-sized parts that will fit in the RAM
                    if len(fit_mode.split('_')) == 2:
                        part_lenght = 150
                    elif len(fit_mode.split('_')) == 3:
                        part_lenght = int(fit_mode.split('_')[-1])
                    self.fit_from_cov(X[train, :], y[train, :],
                                      overwrite=True, part_length=part_lenght)
                else:  # Fit directly -> slightly faster, but uses more RAM
                    self.fit(X[train, :], y[train, :])

            else:
                self.coef_ = self.coef_additive

            if segment_length:  # Chop testing data into smaller pieces

                if (len(test) % segment_length) > 0:  # Crop if there are some odd samples
                    test_crop = test[:-int(len(test) % segment_length)]
                else:
                    test_crop = test[:]

                # Reshape to # segments x segment duration
                test_segments = test_crop.reshape(
                    int(len(test_crop) / segment_length), -1)

                ccs = [self.score(X[test_segments[i], :], y[test_segments[i], :]) for i in range(
                    test_segments.shape[0])]  # Evaluate each segment

                scores.append(ccs)
            else:  # Evaluate using the entire testing data
                scores[kfold, :] = self.score(X[test, :], y[test, :])

        if segment_length:
            scores = np.asarray(scores)

        if train_full:
            if verbose:
                print("Fitting full model...")
            # Fit using trick with adding covariance matrices -> saves RAM
            if fit_mode.find('from_cov') > -1:
                self.fit_from_cov(X, y, overwrite=True,
                                  part_length=part_lenght)
            else:
                self.fit(X, y)

        return scores

    def __getitem__(self, feats):
        "Extract a sub-part of TRF instance as a new TRF instance (useful for plotting only some features...)"
        # Argument check
        if self.feat_names_ is None:
            if np.ndim(feats) > 0:
                assert isinstance(
                    feats[0], int), "Type not understood, feat_names are ot defined, can only index with int"
                indices = feats

            else:
                assert isinstance(
                    feats, int), "Type not understood, feat_names are ot defined, can only index with int"
                indices = [feats]
                feats = [feats]
        else:
            if np.ndim(feats) > 0:
                assert all([f in self.feat_names_ for f in feats]
                           ), "an element in argument %s in not present in %s" % (feats, self.feat_names_)
                indices = [self.feat_names_.index(f) for f in feats]
            else:
                assert feats in self.feat_names_, "argument %s not present in %s" % (
                    feats, self.feat_names_)
                indices = [self.feat_names_.index(feats)]
                feats = [feats]

        trf = TRFEstimator(tmin=self.tmin, tmax=self.tmax,
                           srate=self.srate, alpha=self.alpha)
        trf.coef_ = self.coef_[:, indices]
        trf.feat_names_ = feats
        trf.n_feats_ = len(feats)
        trf.n_chans_ = self.n_chans_
        trf.fitted = True
        trf.times = self.times
        trf.lags = self.lags
        trf.intercept_ = self.intercept_

        return trf

    def __repr__(self):
        obj = """TRFEstimator(
            alpha=%s,
            fit_intercept=%s,
            srate=%s,
            tmin=%s,
            tmax=%s,
            n_feats=%s,
            n_chans=%s,
            n_lags=%s,
            features : %s
        )
        """ % (self.alpha, self.fit_intercept, self.srate, self.tmin, self.tmax,
               self.n_feats_, self.n_chans_, len(self.lags) if self.lags is not None else None, str(self.feat_names_))
        return obj

    def __add__(self, other_trf):
        "Make available the '+' operator. Will simply add coefficients. Be mindful of dividing by the number of elements later if you want the true mean."
        assert (other_trf.n_feats_ == self.n_feats_ and other_trf.n_chans_ ==
                self.n_chans_), "Both TRF objects must have the same number of features and channels"
        trf = TRFEstimator(tmin=self.tmin, tmax=self.tmax,
                           srate=self.srate, alpha=self.alpha)
        trf.coef_ = np.sum([self.coef_, other_trf.coef_], 0)
        trf.intercept_ = np.sum([self.intercept_, other_trf.intercept_], 0)
        trf.feat_names_ = self.feat_names_
        trf.n_feats_ = self.n_feats_
        trf.n_chans_ = self.n_chans_
        trf.fitted = True
        trf.times = self.times
        trf.lags = self.lags

        return trf

    # Obsolete?

    def plot(self, feat_id=None, alpha_id=None, ax=None, spatial_colors=False, info=None, **kwargs):
        """Plot the TRF of the feature requested as a *butterfly* plot.
        Parameters
        ----------
        feat_id : list or int
            Index of the feature requested or list of features.
            Default is to use all features.
        ax : array of axes (flatten)
            list of subaxes
        **kwargs : **dict
            Parameters to pass to :func:`plt.subplots`
        Returns
        -------
        fig : figure
        """
        if isinstance(feat_id, int):
            # cast into list to be able to use min, len, etc...
            feat_id = list(feat_id)
            if ax is not None:
                fig = ax.figure
        if not feat_id:
            feat_id = range(self.n_feats_)
        if len(feat_id) > 1:
            if ax is not None:
                fig = ax[0].figure
        assert self.fitted, "Fit the model first!"
        assert all([min(feat_id) >= 0, max(feat_id) <
                    self.n_feats_]), "Feat ids not in range"

        if ax is None:
            if 'figsize' not in kwargs.keys():
                fig, ax = plt.subplots(nrows=1, ncols=np.size(feat_id), figsize=(
                    plt.rcParams['figure.figsize'][0] * np.size(feat_id), plt.rcParams['figure.figsize'][1]), **kwargs)
            else:
                fig, ax = plt.subplots(
                    nrows=1, ncols=np.size(feat_id), **kwargs)

        if spatial_colors:
            assert info is not None, "To use spatial colouring, you must supply raw.info instance"
            colors = get_spatial_colors(info)

        for k, feat in enumerate(feat_id):
            if len(feat_id) == 1:
                ax.plot(self.times, self.coef_[:, feat, :])
                if self.feat_names_:
                    ax.set_title('TRF for {:s}'.format(self.feat_names_[feat]))
                if spatial_colors:
                    lines = ax.get_lines()
                    for kc, l in enumerate(lines):
                        l.set_color(colors[kc])
            else:
                ax[k].plot(self.times, self.get_coef()[:, feat, :, 0])
                if self.feat_names_:
                    ax[k].set_title('{:s}'.format(self.feat_names_[feat]))
                if spatial_colors:
                    lines = ax[k].get_lines()
                    for kc, l in enumerate(lines):
                        l.set_color(colors[kc])

        return fig


class CCA_Estimator(BaseEstimator):

    """Canonical Correlation (CCA) Estimator Class.

    Attributes
    ----------
    xlags : 1d-array
        Array of `int`, corresponding to lag in samples at which the TRF coefficients are computed
    times : 1d-array
        Array of `float`, corresponding to lag in seconds at which the TRF coefficients are computed
    srate : float
        Sampling rate
    fit_intercept : bool
        Whether a column of ones should be added to the design matrix to fit an intercept
    intercept_ : 1d array (nchans, )
        Intercepts
    coef_ : ndarray (nlags, nfeats, nchans)
        Actual TRF coefficients
    n_feats_ : int
        Number of word level features in TRF
    n_chans_: int
        Number of EEG channels in TRF
    feat_names_ : list
        Names of each word level features
    Notes
    -----
    Attributes with a `_` suffix are only set once the TRF has been fitted on EEG data

   """

    def __init__(self, times=(0.,), tmin=None, tmax=None, filterbank=False, freqs=(0.,), srate=1., fit_intercept=True):

        self.srate = srate
        if filterbank:
            self.f_bank = create_filterbank(
                freqs=freqs, srate=self.srate, N=2, rs=3)

        else:
            if tmin and tmax:
                LOGGER.info(
                    "Will use xlags spanning form tmin to tmax.\nTo use individual xlags, use the `times` argument...")
                self.xlags = lag_span(tmin, tmax, srate=srate)[::-1]
                self.xtimes = self.xlags[::-1] / srate
            else:
                self.xtimes = np.asarray(times)
                self.xlags = lag_sparse(self.xtimes, srate)[::-1]

        self.fit_intercept = fit_intercept
        self.fitted = False
        # All following attributes are only defined once fitted (hence the "_" suffix)
        self.intercept_ = None
        self.coefStim_ = None
        self.coefResponse_ = None
        self.score_ = None
        self.lag_y = False
        self.ylags = None
        self.ytimes = None
        self.eigvals_x = None
        self.eigvals_y = None
        self.n_feats_ = None
        self.n_chans_ = None
        self.feat_names_ = None
        self.sklearn_TRF_ = None
        self.f_bank_freqs = freqs
        self.f_bank_used = filterbank
        self.tempX_path_ = None
        self.tempy_path_ = None

    def fit(self, X, y, cca_implementation='nt', thresh_x=None, normalise=True, thresh_y=None, n_comp=2, knee_point=None, drop=True, y_already_dropped=False, lag_y=False, ylags=(0.,), feat_names=(), opt_cca_svd={}):
        """ Fit CCA model.

        X : ndarray (nsamples x nfeats)
            Array of features (time-lagged)
        y : ndarray (nsamples x nchans)
            EEG data

        Returns
        -------
        coef_ : ndarray (nlags x nfeats)
        intercept_ : ndarray (nfeats x 1)
        """
        if isinstance(y, list):
            print('y is a list. CCA will be implemented more efficiently.')
            self.n_chans_ = y[0].shape[1]
        else:
            self.n_chans_ = y.shape[1]
        self.n_feats_ = X.shape[1]
        if feat_names:
            self.feat_names_ = feat_names

        # Creating filterbank
        if self.f_bank_used:
            temp_X = apply_filterbank(X, self.f_bank)
            X = np.reshape(
                temp_X, (X.shape[0], temp_X.shape[0]*temp_X.shape[2]))
            if isinstance(y, list):
                filterbank_y = []
                for subj in range(len(y)):
                    # NEED TO CHANGE TO drop_missing=True
                    temp = apply_filterbank(y[subj], self.f_bank)
                    temp_y = np.reshape(
                        temp, (y[subj].shape[0], temp.shape[0]*temp.shape[2]))
                    filterbank_y.append(temp_y)
            else:
                # NEED TO CHANGE TO drop_missing=True
                temp = apply_filterbank(y, self.f_bank)
                filterbank_y = np.reshape(
                    temp, (y.shape[0], temp.shape[0]*temp.shape[2]))
            y = filterbank_y
        else:
            # Creating lag-matrix droping NaN values if necessary
            if drop:
                X = lag_matrix(X, lag_samples=self.xlags, drop_missing=True)

                if not y_already_dropped:
                    # Droping rows of NaN values in y
                    if isinstance(y, list):
                        temp = []
                        for yy in y:
                            if any(np.asarray(self.xlags) < 0):
                                drop_top = abs(min(self.xlags))
                                yy = yy[drop_top:,
                                        :] if yy.ndim == 2 else yy[:, drop_top:, :]
                            if any(np.asarray(self.xlags) > 0):
                                drop_bottom = abs(max(self.xlags))
                                yy = yy[:-drop_bottom,
                                        :] if yy.ndim == 2 else yy[:, :-drop_bottom, :]
                            temp.append(yy)
                        y = temp
                    else:
                        if any(np.asarray(self.xlags) < 0):
                            drop_top = abs(min(self.xlags))
                            y = y[drop_top:, :] if y.ndim == 2 else y[:,
                                                                      drop_top:, :]
                        if any(np.asarray(self.xlags) > 0):
                            drop_bottom = abs(max(self.xlags))
                            y = y[:-drop_bottom,
                                  :] if y.ndim == 2 else y[:, :-drop_bottom, :]

                if lag_y:
                    self.lag_y = True
                    self.ytimes = np.asarray(ylags)
                    self.ylags = -lag_sparse(self.ytimes, self.srate)[::-1]
                    if isinstance(y, list):
                        lagged_y = []
                        for subj in range(len(y)):
                            # NEED TO CHANGE TO drop_missing=True
                            temp = lag_matrix(
                                y[subj], lag_samples=self.ylags, drop_missing=False, filling=0.)
                            lagged_y.append(temp)
                    else:
                        # NEED TO CHANGE TO drop_missing=True
                        lagged_y = lag_matrix(
                            y, lag_samples=self.ylags, drop_missing=False, filling=0.)
                        print(lagged_y.shape)
                    y = lagged_y
            else:
                X = lag_matrix(X, lag_samples=self.xlags, filling=0.)
                if lag_y:
                    self.lag_y = True
                    self.ytimes = np.asarray(ylags)
                    self.ylags = -lag_sparse(self.ytimes, self.srate)[::-1]
                    if isinstance(y, list):
                        lagged_y = []
                        for subj in range(len(y)):
                            temp = lag_matrix(
                                y[subj], lag_samples=self.ylags, filling=0.)
                            lagged_y.append(temp)
                    else:
                        lagged_y = lag_matrix(
                            y, lag_samples=self.ylags, filling=0.)
                    y = lagged_y

        # Adding intercept feature:
        if self.fit_intercept:
            X = np.hstack([np.ones((len(X), 1)), X])
        if thresh_x is None:
            if thresh_y is None:
                thresh_x = 0.999
                thresh_y = 0.999
            else:
                thresh_x = thresh_y
        if thresh_y is None:
            thresh_y = thresh_x
        threshs = np.asarray([thresh_x, thresh_y])

        if cca_implementation == 'nt':
            A1, A2, A, B, R, eigvals_x, eigvals_y = cca_nt(
                X, y, threshs, knee_point)
            # Reshaping and getting coefficients
            if self.fit_intercept:
                self.intercept_ = A[0, :]
                A = A[1:, :]

            self.coefResponse_ = B
            self.score_ = R
            self.eigvals_x = eigvals_x
            self.eigvals_y = eigvals_y

        if cca_implementation == 'svd':
            Ax, Ay, R = cca_svd(X, y, opt_cca_svd)
            # Reshaping and getting coefficients
            if self.fit_intercept:
                self.intercept_ = Ax[0, :]
                A = Ax[1:, :]
            else:
                A = Ax

            self.coefResponse_ = Ay
            self.score_ = R

        if cca_implementation == 'sklearn':
            cca_skl = CCA(n_components=n_comp)
            cca_skl.fit(X, y)
            A = cca_skl.x_rotations_
            if self.fit_intercept:
                self.intercept_ = A[0, :]
                A = A[1:, :]

            self.coefResponse_ = cca_skl.y_rotations_
            score = np.diag(np.corrcoef(cca_skl.x_scores_,
                                        cca_skl.y_scores_, rowvar=False)[:n_comp, n_comp:])
            self.score_ = score
            self.sklearn_TRF_ = cca_skl.coef_

        # save the matrix X and y to save memory
        if self.fit_intercept:
            X = X[:, 1:]
        if sys.platform.startswith("win"):
            tmpdir = os.environ["TEMP"]
        else:
            #tmpdir = os.environ["TMPDIR"]
            tmpdir = '/home/phg17/Documents/Personal/EEG - Octave/Temporary'
        np.save(os.path.join(tmpdir, 'temp_X'), X)
        if isinstance(y, list):
            np.save(os.path.join(tmpdir, 'temp_y'), np.asarray(
                y).reshape((len(y)*y[0].shape[0], y[0].shape[1])))
        else:
            np.save(os.path.join(tmpdir, 'temp_y'), y)
        self.tempX_path_ = os.path.join(tmpdir, 'temp_X')
        self.tempy_path_ = os.path.join(tmpdir, 'temp_y')

        if self.f_bank_used:
            self.coefStim_ = np.reshape(
                A, (len(self.f_bank_freqs), self.n_feats_, self.coefResponse_.shape[1]))
        else:
            self.coefStim_ = np.reshape(
                A, (len(self.xlags), self.n_feats_, self.coefResponse_.shape[1]))
            self.coefStim_ = self.coefStim_[::-1, :, :]

    def transform(self, transform_x=True, transform_y=False, comp=0):
        """ Transform X and Y using the coefficients
        """
        X = np.load(self.tempX_path_+'.npy')
#        y = np.load(self.tempy_path_+'.npy')
#        if len(y) > len(X):
#            all_x = np.concatenate([X for i in range(int(len(y)/len(X)))])
#        else:
#            all_x = X
#        coefStim_ = self.coefStim_.reshape((self.coefStim_.shape[0] * self.coefStim_.shape[1], self.coefStim_.shape[2]))
#
#        if transform_x:
#            return all_x @ coefStim_[:, comp]
#        if transform_y:
#            return y @ self.coefResponse_[:, comp]
        return self.coefResponse_.T @ self.coefStim_.T @ X

    def plot_time_filter(self, n_comp=1, dim=[0]):
        """Plot the TRF of the feature requested.
        Parameters
        ----------
        feat_id : int
            Index of the feature requested
        """
        if n_comp < 6:
            for c in range(n_comp):
                for d in range(len(dim)):
                    plt.plot(self.xtimes, self.coefStim_[
                             :, dim[d], c], label='CC #%s, dim: %s' % ((c+1), dim[d]))
        else:
            for c in range(5):
                for d in range(len(dim)):
                    plt.plot(self.xtimes, self.coefStim_[
                             :, dim[d], c], label='CC #%s, dim: %s' % ((c+1), dim[d]))
            for c in range(5, n_comp):
                for d in range(len(dim)):
                    plt.plot(self.xtimes, self.coefStim_[:, dim[d], c])
        if self.feat_names_:
            plt.title('Time filter for {:s}'.format(self.feat_names_[0]))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('Time (s)')
        plt.ylim([-max(np.abs(self.coefStim_[:, dim, :n_comp].flatten())),
                  max(np.abs(self.coefStim_[:, dim, :n_comp].flatten()))])

    def plot_spatial_filter(self, pos, n_comp=1):
        """Plot the topo of the feature requested.
        Parameters
        ----------
        feat_id : int
            Index of the feature requested
        """
        titles = [r"CC #{:d}, $\rho$={:.3f} ".format(
            k+1, c) for k, c in enumerate(self.score_)]
        topoplot_array(self.coefResponse_, pos, n_topos=n_comp, titles=titles)
        mne.viz.tight_layout()

    def plot_corr(self, pos, n_comp=1):
        """Plot the correlation between the EEG component waveform and the EEG channel waveform.
        Parameters
        ----------
        """
        X = np.load(self.tempX_path_+'.npy')
        y = np.load(self.tempy_path_+'.npy')
        if len(y) > len(X):
            all_x = np.concatenate([X for i in range(int(len(y)/len(X)))])
        else:
            all_x = X
        coefStim_ = self.coefStim_.reshape(
            (self.coefStim_.shape[0] * self.coefStim_.shape[1], self.coefStim_.shape[2]))

        r = np.zeros((64, n_comp))
        for c in range(n_comp):
            eeg_proj = y @ self.coefResponse_[:, c]
            env_proj = all_x @ coefStim_[:, c]
            for i in range(64):
                r[i, c] = np.corrcoef(y[:, i], eeg_proj)[0, 1]
            # cc_corr = np.corrcoef(eeg_proj, env_proj)[0,1]

        titles = [r"CC #{:d}, $\rho$={:.3f} ".format(
            k+1, c) for k, c in enumerate(self.score_)]
        topoplot_array(r, pos, n_topos=n_comp, titles=titles)
        mne.viz.tight_layout()

    def plot_activation_map(self, pos, n_comp=1, lag=0):
        """Plot the activation map from the spatial filter.
        Parameters
        ----------
        """
        y = np.load(self.tempy_path_+'.npy')
        if n_comp <= 0:
            print('Invalid number of components, must be a positive integer.')

        s_hat = y @ self.coefResponse_
        sigma_eeg = y.T @ y
        sigma_reconstr = s_hat.T @ s_hat
        a_map = sigma_eeg @ self.coefResponse_ @ np.linalg.inv(sigma_reconstr)

        if self.lag_y | self.f_bank_used:
            if self.f_bank_used:
                a_map = np.reshape(
                    a_map, (len(self.f_bank_freqs), self.n_chans_, self.coefResponse_.shape[1]))
            else:
                a_map = np.reshape(
                    a_map, (self.ylags.shape[0], self.n_chans_, self.coefResponse_.shape[1]))
            titles = [r"CC #{:d}, $\rho$={:.3f} ".format(
                k+1, c) for k, c in enumerate(self.score_)]
            fig = plt.figure(figsize=(12, 10), constrained_layout=False)
            outer_grid = fig.add_gridspec(5, 5, wspace=0.0, hspace=0.25)
            for c in range(n_comp):
                inner_grid = outer_grid[c].subgridspec(1, 1)
                ax = plt.Subplot(fig, inner_grid[0])
                im, _ = mne.viz.plot_topomap(
                    a_map[lag, :, c], pos, axes=ax, show=False)
                ax.set(title=titles[c])
                fig.add_subplot(ax)
            mne.viz.tight_layout()

        else:
            titles = [r"CC #{:d}, $\rho$={:.3f} ".format(
                k+1, c) for k, c in enumerate(self.score_)]
            topoplot_array(a_map, pos, n_topos=n_comp, titles=titles)
            mne.viz.tight_layout()

    def plot_compact_time(self, n_comp=2, dim=0):
        plt.imshow(self.coefStim_[:, dim, :n_comp].T, aspect='auto', origin='bottom', extent=[
                   self.xtimes[0], self.xtimes[-1], 0, n_comp])
        plt.colorbar()
        plt.ylabel('Components')
        plt.xlabel('Time (ms)')
        plt.title('Dimension #{:d}'.format(dim+1))

    def plot_all_dim_time(self, n_comp=0, n_dim=2):
        n_comp = range(n_comp)
        n_rows = len(n_comp) // 2 + len(n_comp) % 2
        fig = plt.figure(figsize=(10, 20), constrained_layout=False)
        outer_grid = fig.add_gridspec(n_rows, 2, wspace=0.1, hspace=0.1)
        bottoms, tops, _, _ = outer_grid.get_grid_positions(fig)
        for c, coefs in enumerate(self.coefStim_.swapaxes(0, 2)[n_comp]):
            inner_grid = outer_grid[c].subgridspec(1, 1)
            ax = plt.Subplot(fig, inner_grid[0])
            vmin = np.min(coefs)
            vmax = np.max(coefs)
            im = ax.imshow(coefs, aspect=0.04, origin='bottom',
                           extent=[self.xtimes[0], self.xtimes[-1], 0, n_dim],
                           vmin=vmin, vmax=vmax)
            if c // 2 != n_rows-1:
                ax.set_xticks([])
            else:
                ax.set_xlabel('Time (s)')
            if c % 2 != 0:
                ax.set_yticks([])
            else:
                ax.set_ylabel('Dimension')
            ax.set(title=('CC #{:d}'.format(c+1)))
            fig.add_subplot(ax)
        # Colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.5 - 0.05, 0.01, 0.1])
        fig.colorbar(im, cax=cbar_ax, shrink=0.6)
