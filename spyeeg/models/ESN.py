"""
Simple implementation of an Echo State Network (ESN), using reservoirpy:
Nathan Trouvain, Luca Pedrelli, Thanh Trung Dinh, Xavier Hinaut. 
ReservoirPy: an Efficient and User-Friendly Library to Design Echo State Networks. 
2020. ⟨hal-02595026⟩ https://hal.inria.fr/hal-02595026

This function is mostly a wrapper for the most simple usecase of ESN (no feedback, single reservoir).
It thus can be compared to a TRF with various non-linnear dynamics rather than timelags being applied on the feature matrix.
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
from matplotlib import colormaps as cmaps
from sklearn.preprocessing import scale

MEM_CAP = 0.9  # Memory cap for the iRRR model (in GB)

class ESNEstimator(BaseEstimator):

    def __init__(self, srate, alpha = [0], n_units = 500, sr = 0.9, lr = 0.5, 
                 scale_reservoir = False, percentile_units = 0, separate_features = False):
        '''
        Echo State Network, no initialization
        '''
        # General parameters
        self.srate = srate
        self.fitted = False
        self.alpha = alpha
        self.n_units = n_units
        self.sr = sr
        self.lr = lr
        self.scale_reservoir = scale_reservoir
        self.percentile_units = percentile_units
        self.separate_features = separate_features
        
        # All following attributes are only defined once fitted (hence the "_" suffix)
        self.intercept_ = None
        self.coef_ = None
        self.reservoir_activity = None
        self.n_feats_ = None
        self.n_chans_ = None
        self.n_samples_ = None


        # Generate the reservoir
        self.reservoir = Reservoir(n_units, lr=lr, sr=sr)

    def run_reservoir(self, X):
        '''
        Run the input data into the reservoir.
        ----------
        X : ndarray (T x nfeat)
        scale_reservoir : bool, whether to scale the output of the reservoir
        separate_features : bool, whether to run a separate reservoir on each feature. Careful, this increases the total number of units.
        '''
        if self.separate_features:
            X_reservoir = np.zeros([X.shape[0], X.shape[1] * self.n_units])
            for i_feat in range(X.shape[1]):
                X_reservoir[:,i_feat*self.n_units:(i_feat+1)*self.n_units] = self.reservoir.run(X[:,i_feat:(i_feat+1)], reset=True)
            X = X_reservoir
        else:
            X = self.reservoir.run(X, reset=True)
        if self.scale_reservoir:
            X = scale(X,axis=0)
        if self.separate_features:
            True
        self.reservoir_activity = X.copy()
        return X

    def get_XY(self, X, y):
        '''
        Preprocess X and y before fitting (finding mapping between X -> y)
        Parameters
        ----------
        X : ndarray (T x nfeat)
        y : ndarray (T x nchan)
        Returns
        -------
        Features preprocessed for fitting the model.
        X : ndarray (T x n_units)
        y : ndarray (T x nchan)
        '''

        X = np.asarray(X)
        y = np.asarray(y)

        # Estimate the necessary size to compute stuff
        y_memory = sum([yy.nbytes for yy in y]) if np.ndim(
            y) == 3 else y.nbytes
        estimated_mem_usage = X.nbytes * \
            (self.n_units) + y_memory
        if estimated_mem_usage/1024.**3 > MEM_CAP*mem_check():
            raise MemoryError("Not enough RAM available! (needed %.1fGB, but only %.1fGB available)" % (
                estimated_mem_usage/1024.**3, mem_check()))

        # Fill n_feat and n_chan attributes
        self.n_feats_ = X.shape[1]
        self.n_chans_ = y.shape[1] if y.ndim == 2 else y.shape[2]

        # this include non-valid samples for now
        n_samples_all = y.shape[0] if y.ndim == 2 else y.shape[1]

        # Run X through Reservoir. The reservoir is reset everytime to avoid data leaking.
        # This step is redone at each xvalidation to avoid data leaking from training to testing via long-term dynamics.
        X = self.run_reservoir(X)

        return X, y

    def fit(self, X, y):
        """Fit the TRF model.
        Mapping X -> y. Note the convention of timelags and type of model for seamless recovery of coefficients.
        Parameters
        ----------
        X : ndarray (nsamples x nfeats)
        y : ndarray (nsamples x nchans)
        Returns
        -------
        coef_ : ndarray (alphas x nlags x nfeats)
        intercept_ : ndarray (nfeats x 1)
        """

        # Preprocess and compute reservoir activity
        X, y = self.get_XY(X, y)

        # Regress with Ridge to obtain coef for the input alpha
        self.coef_ = _ridge_fit_SVD(X, y, self.alpha, alpha_feat = False)
        self.fitted = True
        return self

    def get_coef(self):
        '''
        Format and return coefficients. Note mtype attribute needs to be declared in the __init__.

        Returns
        -------
        coef_ : ndarray (n_units x nchans x regularization params)
        '''
        if np.ndim(self.alpha) == 0:
            betas = np.reshape(self.coef_, (self.n_units, self.n_chans_))
        else:
            betas = np.reshape(self.coef_, (self.n_units, self.n_chans_, len(self.alpha)))
        return betas

    def predict(self, X):
        """Compute output based on fitted coefficients and feature matrix X.
        Parameters
        ----------
        X : ndarray
            Matrix of features
        Returns
        -------
        ndarray
            Reconstruction of target with current beta estimates
        """
        assert self.fitted, "Fit model first!"

        betas = self.coef_[:]
        if self.percentile_units != 0:
            betas[np.abs(betas) < np.percentile(np.abs(betas), self.percentile_units)] = 0
        X = self.run_reservoir(X)

        # Do it for every alpha
        pred = np.stack([X.dot(betas[..., i])
                         for i in range(betas.shape[-1])], axis=-1)

        return pred  # Shape T x Nchan x Alpha

    def score(self, Xtest, ytrue, Xtrain = None, scoring="R2"):
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
        yhat = self.predict(Xtest)
        reg_len = len(self.alpha)
        alpha = self.alpha
        if scoring == 'corr':
            scores = np.stack([_corr_multifeat(yhat[..., a], ytrue, nchans=self.n_chans_) for a in range(reg_len)], axis=-1)
            self.scores = scores
            return scores
        elif scoring == 'rmse':
            scores = np.stack([_rmse_multifeat(yhat[..., a], ytrue) for a in range(reg_len)], axis=-1)
            self.scores = scores
            return scores
        elif scoring == 'R2':
            scores = np.stack([_r2_multifeat(yhat[..., a], ytrue) for a in range(reg_len)], axis=-1)
            self.scores = scores
            return scores
        elif scoring == 'rankcorr':
            scores = np.stack([_rankcorr_multifeat(yhat[..., a], ytrue, nchans=self.n_chans_) for a in range(reg_len)], axis=-1)
            self.scores = scores
            return scores
        elif scoring == 'ezekiel':
            scores = np.stack([_ezr2_multifeat(yhat[..., a], ytrue, Xtest, window_length = 1) for a in range(reg_len)], axis=-1)
            self.scores = scores
            return scores
        elif scoring == 'adj_R2':
            scores = np.stack([_adjr2_multifeat(yhat[..., a], ytrue, Xtrain, Xtest, alpha[a], [0]) for a in range(reg_len)], axis=-1)
            self.scores = scores
            return scores
        else:
            raise NotImplementedError(
                "Only correlation & RMSE scores are valid for now...")

    def xval_eval(self, X, y, n_splits=5, lagged=False, drop=True, train_full=True, scoring="R2", segment_length=None, fit_mode='direct', verbose=True):
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

        self.n_feats_ = X.shape[1]
        self.n_chans_ = y.shape[1] if y.ndim == 2 else y.shape[2]

        reg_len = len(self.alpha)

        kf = KFold(n_splits=n_splits)
        if segment_length:
            scores = []
        else:
            scores = np.zeros((n_splits, self.n_chans_, reg_len))

        for kfold, (train, test) in enumerate(kf.split(X)):
            if verbose:
                print("Training/Evaluating fold %d/%d" % (kfold+1, n_splits))

            self.fit(X[train, :], y[train, :])
            scores[kfold, :] = self.score(X[test, :], y[test, :], scoring=scoring, Xtrain = X[train, :])

        if train_full:
            if verbose:
                print("Fitting full model...")
            # Fit using trick with adding covariance matrices -> saves RAM
                self.fit(X, y)
        self.scores = scores

        return scores




    def get_best_alpha(self):
        best_alpha = np.zeros(self.n_chans_)
        for chan in range(self.n_chans_):
            if len(self.scores.shape) == 3:
                best_alpha[chan] = np.argmax(np.mean(self.scores[:,chan,:],axis=0))
            else:
                best_alpha[chan] = np.argmax(self.scores[:,chan,:],axis=0)
        return best_alpha.astype(int)


    def plot_score(self, figax = None, figsize = (5,5), color_type = 'rainbow', 
                   channels = [], title = 'R2 sumary', minR2 = -np.inf):
        if figax == None:
            fig,ax = plt.subplots(figsize = figsize)
        else:
            fig,ax = figax
        if len(channels) == 0:
            channels = np.arange(self.scores.shape[1])

        #Extract Coef
        color_map = dict()
        for index_channel in range(self.scores.shape[1]):
            color_map[index_channel] = cmaps[color_type](index_channel/self.scores.shape[1])

        for index_channel in range(self.scores.shape[1]):
            score_chan = np.mean(self.scores[:,index_channel,:],axis = 0)
            if np.max(score_chan > minR2):
                ax.plot(self.alpha, score_chan, color = color_map[index_channel], linewidth = 1.5, label = channels[index_channel])
        ax.set_title(title)
        ax.set_xlabel('Alpha')
        ax.set_ylabel('R2')
        ax.set_xticks(self.alpha)
        ax.set_xscale('log')
        ax.plot(self.alpha, np.mean(self.scores[:,:,:],axis = (0,1)), color = 'k', linewidth = 3, linestyle = '--')
        ax.legend()

        return fig, ax

    def __repr__(self):
        obj = """TRFEstimator(
            alpha=%s,
            srate=%s,
            n_feats=%s,
            n_chans=%s,
            n_units=%s
        )
        """ % (self.alpha, self.srate,
               self.n_feats_, self.n_chans_, self.n_units)
        return obj

    def __add__(self, other_trf):
        "Make available the '+' operator. Will simply add coefficients. Be mindful of dividing by the number of elements later if you want the true mean."
        assert (other_trf.n_feats_ == self.n_feats_ and other_trf.n_chans_ ==
                self.n_chans_), "Both TRF objects must have the same number of features and channels"
        trf = TRFEstimator(tmin=self.tmin, tmax=self.tmax,
                           srate=self.srate, alpha=self.alpha)
        trf.coef_ = np.sum([self.coef_, other_trf.coef_], 0)
        trf.n_feats_ = self.n_feats_
        trf.n_chans_ = self.n_chans_
        trf.fitted = True
        trf.times = self.times
        trf.lags = self.lags

        return trf

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

