"""
Simple implementation of an Echo State Network, using reservoirpy:
Nathan Trouvain, Luca Pedrelli, Thanh Trung Dinh, Xavier Hinaut. 
ReservoirPy: an Efficient and User-Friendly Library to Design Echo State Networks. 
2020. ⟨hal-02595026⟩ https://hal.inria.fr/hal-02595026
"""


import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from mne.decoding import BaseEstimator
from ..utils import lag_matrix, lag_span, lag_sparse, mem_check, get_timing, center_weight, count_significant_figures
from ..viz import get_spatial_colors
from scipy import linalg
import mne
from numpy.random import randn
from ._methods import _ridge_fit_SVD, _get_covmat, _corr_multifeat, _rmse_multifeat, _objective_value, _soft_threshold, _r2_multifeat
from reservoirpy.nodes import Reservoir, Ridge, Input, Concat
from reservoirpy.observables import rmse, rsquare
from scipy.linalg import pinv, svd, norm, svdvals

MEM_CAP = 0.9  # Memory cap for the iRRR model (in GB)

class ESNEstimator(BaseEstimator):

    def __init__(self, times=(0.,), alpha = [0], units = 500, feedback = False):
        '''
        Echo State Network, no initialization
        '''
        # General parameters
        self.tmin = tmin
        self.tmax = tmax
        self.times = times
        self.srate = srate
        self.fitted = False
        self.lags = None

        # All following attributes are only defined once fitted (hence the "_" suffix)
        self.intercept_ = None
        self.coef_ = None
        self.n_feats_ = None
        self.n_chans_ = None
        self.n_samples_ = None
        self.n_featlags_ = None
        self.feat_names_ = None
        self.valid_samples_ = None
        self.n_featlags_cumsum_ = None

        # Autocorrelation matrix of feature X (thus XtX) -> used for computing model using fit_from_cov
        self.XtX_ = None
        # Covariance matrix of features X and Y (thus XtX) -> used for computing model using fit_from_cov
        self.XtY_ = None

        # Fit Attributes
        self.weight = None
        self.randomstart = None
        self.varyrho = None
        self.maxrho = None
        self.rho = None
        self.tol = None
        self.Niter = None
        self.pre_fit = None

        # Coefficients of the iRRR
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lowrankcoef_ = None
        self.highrankcoef_ = None
        self.stackcoef_ = None
        self.intercept_ = None
        self.Theta_ = None
        self.niter_ = None
        self.rec_Theta_ = None
        self.rec_nonzeros_ = None
        self.rec_primal_ = None
        self.rec_dual_ = None
        self.rec_obj_ = None



    def fit(self, X, y, record_state = False, feedback = False, verbose = False):
        """
        Mapping X -> y
        Parameters
        ----------
        y: ndarray (nsamples x nchans)
            The dependent variable (EEG data)
        X: ndarray (nsamples x nfeats)
            The regressors. Similarly to the TRF, the same regularization parameter is used for different features for now.
            But scale your regressors for now.
        lambda1: float (positive)
            Regularization parameter, tuning for nuclear norm
        lambda0: float (positive)
            Default 0. Regularization parameter, tuning for ridge penalty
        param_dict: dict
            Dictionary of parameters for the model. If None, default parameters are used.
                weight: ndarray (nfeats x 1) 
                    Default: np.ones(nfeats)
                    weight vector, theoretically w(i)=(1/n)*max(svd(X{i}))*(sqrt(q)+sqrt(rank(X{i})))
                        heuristically, w(i)=|X_i|_F
                    randomstart: bool or list 
                        Default: False (bool)
                        When True, B is initialized using random values. When False, B is initialized using vanilla OLS.
                        If B is a list, we explicitely the values inside as initial condition for B.
                    varyrho: bool
                        Default: False (bool)
                        Whether or not rho should be adaptative
                    maxrho: float 
                        Default: 5 
                        Maximum value of rho, unused if varyrho==0
                    rho: float
                        Default: 0.1
                        Initial step size
                    Tol: float
                        Default: 1E-3
                        Error tolerance
                    Niter: int
                        Default: 500
                        Number of iterations

        Returns
        -------
        self : instance of iRRREstimator
            The iRRR instance.

        """
        return True

    def get_coef(self):
        '''
        Format and return coefficients. 

        Returns
        -------
        coef_ : ndarray (nlags x nfeats x nchans x regularization params)
        '''

        betas_high = self.highrankcoef_.swapaxes(0,1)[::-1, :]
        betas_low = self.lowrankcoef_.swapaxes(0,1)[::-1, :]

        return betas_high, betas_low
    
    def predict(self, X, lowrank = True):
        """Compute output based on fitted coefficients and feature matrix X.
        Parameters
        ----------
        X : ndarray
            Matrix of features (can be already lagged or not).
        lowrank : str
            Whether to compute with the low or not
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
        if lowrank:
            betas = self.stacklowrankcoef_
        else:
            betas = self.stackhighrankcoef_

        # Check if input has been lagged already, if not, do it:
        if X.shape[1] != len(self.lags) * self.n_feats_:
            X = lag_matrix(X, lag_samples=self.lags, filling=0.)

        # Predict it for every lambda0 and lambda1
        pred_list = [[X.dot(betas[..., i, j]) for j in range(betas.shape[-1])] for i in range(betas.shape[-2])]
        pred = np.transpose(np.array(pred_list), (2,3,0,1))

        return pred  # Shape T x Nchan x n_lambda0 x n_lambda1
    
    def score(self, Xtest, ytrue, scoring="corr", lowrank = True):
        """Compute a score of the model given true target and estimated target from Xtest.
        Parameters
        ----------
        Xtest : ndarray
            Array used to get "yhat" estimate from model
        ytrue : ndarray
            True target
        scoring : str (or func in future?)
            Scoring function to be used ("corr", "rmse", "R2")
        Returns
        -------
        float
            Score value computed on whole segment.
        """
        yhat = self.predict(Xtest, lowrank = lowrank)
        if scoring == 'corr':
            scores_ij = np.array([[_corr_multifeat(yhat[..., i, j], ytrue, nchans=self.n_chans_)
                                for j in range(len(self.lambda1))] for i in range(len(self.lambda0))])
            # Stack along a new last axis, keeping i and j as separate dimensions
            scores = np.transpose(np.stack(scores_ij, axis=-1),(1,2,0))
            self.scores = scores
            # Return this array as the result
            return scores
        
        elif scoring == 'rmse':
            scores_ij = np.array([[_rmse_multifeat(yhat[..., i, j], ytrue, nchans=self.n_chans_)
                                for j in range(len(self.lambda1))] for i in range(len(self.lambda0))])
            # Stack along a new last axis, keeping i and j as separate dimensions
            scores = np.transpose(np.stack(scores_ij, axis=-1),(1,2,0))
            self.scores = scores
            # Return this array as the result
            return scores
        elif scoring == 'R2':
            scores_ij = np.array([[_r2_multifeat(yhat[..., i, j], ytrue)
                                for j in range(len(self.lambda1))] for i in range(len(self.lambda0))])
            # Stack along a new last axis, keeping i and j as separate dimensions
            scores = np.transpose(np.stack(scores_ij, axis=-1),(1,2,0))
            self.scores = scores
            # Return this array as the result
            return scores
        else:
            raise NotImplementedError(
                "Only correlation, R2 & RMSE scores are valid for now...")
        
    def xval_eval(self, X, y, n_splits=5, lagged=False, drop=True, train_full=False, scoring="corr", 
                  lowrank = True, segment_length=None, fit_mode='direct', verbose=True):
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

        if np.ndim(self.lambda0) + np.ndim(self.lambda1) < 1 or len(self.lambda0) + len(self.lambda1) <= 1:
            raise ValueError(
                "Supply several regularization values to TRF constructor to use this method.")

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
            scores = np.zeros((n_splits, self.n_chans_, len(self.lambda0), len(self.lambda1)))

        for kfold, (train, test) in enumerate(kf.split(X)):
            if verbose:
                print("Training/Evaluating fold %d/%d" % (kfold+1, n_splits))

            # Fit using trick with adding covariance matrices -> saves RAM
            self.fit(X[train, :], y[train, :])

            if segment_length:  # Chop testing data into smaller pieces

                if (len(test) % segment_length) > 0:  # Crop if there are some odd samples
                    test_crop = test[:-int(len(test) % segment_length)]
                else:
                    test_crop = test[:]

                # Reshape to # segments x segment duration
                test_segments = test_crop.reshape(
                    int(len(test_crop) / segment_length), -1)

                ccs = [self.score(X[test_segments[i], :], y[test_segments[i], :], lowrank = True, scoring = scoring) for i in range(
                    test_segments.shape[0])]  # Evaluate each segment

                scores.append(ccs)
            else:  # Evaluate using the entire testing data
                scores[kfold, :] = self.score(X[test, :], y[test, :], lowrank = True, scoring = scoring)

        if segment_length:
            scores = np.asarray(scores)

        if train_full:
            if verbose:
                print("Fitting full model...")
            # Fit using trick with adding covariance matrices -> saves RAM
            self.fit(X, y)

        return scores