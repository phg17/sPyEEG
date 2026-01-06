"""
Classic TRF.
"""

import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from mne.decoding import BaseEstimator
from ..utils import lag_matrix, lag_span, lag_sparse, mem_check
from scipy import linalg
import mne
from ._methods import fit_iRRR_fista, fit_RRR_fista
from ._methods import _get_covmat, _corr_multifeat, _rmse_multifeat, _r2_multifeat, _rankcorr_multifeat, _ezr2_multifeat, _adjr2_multifeat
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy.stats import pearsonr
from matplotlib import colormaps as cmaps

class RRREstimator(BaseEstimator):
    """
    This implements a RRR to relate a continuous stimuli and electrophysiological data. The model functions under the same assumptions as the TRF (linear time-invariance), but assumes a low-rank response of the system.

    Parameters
    ----------
    times : tuple
        mismatch a -> b, where a - dependent, b - predicted. Negative timelags indicate a lagging behind b. Positive timelags indicate b lagging behind a
    tmin : float
        Minimum time lag (in seconds). Can be negative to check for lags in the past (~null model).
    tmax : float
        Maximum time lag (in seconds). Can be large to check for lags in the future (~null model).
    srate : float
        Sampling rate of the data.
    lam : list
        Regularization parameter for the model. 
        This regularization parameter represents how much the model seeks to reduce the rank of the representation.
    """

    def __init__(self, times=(0.,), tmin=None, tmax=None, srate=1., lam=0., max_iter=1000, tol=1e-6, blocks = False):
        """
        Initialize the class instance.
        """

        self.tmin = tmin
        self.tmax = tmax
        self.times = times
        self.srate = srate
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.blocks = blocks
        # Forward or backward. Required for formatting coefficients in get_coef (convention: forward - stimulus -> eeg, backward - eeg - stimulus)
        self.fitted = False
        self.lags = None

        # All following attributes are only defined once fitted (hence the "_" suffix)
        self.intercept_ = None
        self.coef_ = None
        self.n_feats_ = None
        self.n_chans_ = None
        self.feat_names_ = None
        self.valid_samples_ = None
        # Scores when computed
        self.scores = None
        self.rank = None

    def fill_lags(self):
        """
        Fill the lags attributes, with number of samples and times in seconds.
        """
        if (self.tmin != None) and (self.tmax != None):
            # LOGGER.info("Will use lags spanning form tmin to tmax.\nTo use individual lags, use the `times` argument...")
            self.lags = lag_span(self.tmin, self.tmax, srate=self.srate)[
                ::-1]  # pylint: disable=invalid-unary-operand-type
            # self.lags = lag_span(-tmax, -tmin, srate=srate) #pylint: disable=invalid-unary-operand-type
            self.times = self.lags[::-1] / self.srate
        else:
            self.times = np.asarray(self.times)
            self.lags = lag_sparse(self.times, self.srate)[::-1]

    def get_XY(self, X, y, lagged=False, drop=True, feat_names=()):
        """
        Preprocess X and y before fitting (finding mapping between X -> y).
        
        Parameters
        ----------
        X : ndarray 
            input, of shape (T, nfeat)
        y : ndarray 
            output, of shape (T, nchan)
        lagged : bool
            Whether the X matrix has been previously 'lagged'.
        drop : bool
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        feat_names : list
            Names of features being fitted. Must be of length ``nfeats``.
            
        Returns
        -------
        X : ndarray 
            Preprocessed input, of shape (T, nlags * nfeats)
        y : ndarray
            Preprocessed output, of shape (T, nlags * nfeats)
        """
        self.fill_lags()

        X = np.asarray(X)
        y = np.asarray(y)

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
        """
        Fit the RRR model in either the time of frequency domain. The convention is to map X -> y. In order to properly retrieve the coefficients, respect the convention using the 'mtype' argument. This fills the coefficients attributes with shape (alphas, nlags, nfeats)
        
        Parameters
        ----------
        X : ndarray 
            input of shape (T, nfeats)
        y : ndarray
            output of shape (T, nfeats)
        lagged : bool
            Whether the X matrix has been previously 'lagged'.
        drop : bool
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        feat_names : list
            Names of features being fitted. Must be of length ``nfeats``.
            
        Returns
        -------
        coef_ : ndarray 
            coefficients of shape (alphas, nlags, nfeats)
        """

        # Preprocess and lag inputs
        X, y = self.get_XY(X, y, lagged, drop, feat_names)

        # Regress with Ridge to obtain coef for the input alpha
        if not self.blocks:
            self.coef_, info_RRR = fit_RRR_fista(X, y, lam = self.lam, max_iter=self.max_iter, tol= self.tol)
        else:
            X_blocks = X.reshape(X.shape[0],len(self.times),self.n_feats_)
            X_blocks = list(np.moveaxis(X_blocks, 2, 0))
            self.coef_, info_RRR = fit_iRRR_fista(X_blocks, y, lam_blocks=self.lam, verbose=False, max_iter=1000)
        self.fitted = True
        self.rank = info_RRR['rank'][-1]

        return self.coef_.copy()

    def get_coef(self):
        """
        Format and return coefficients. Note mtype attribute needs to be declared in the __init__.

        Returns
        -------
        coef_ : ndarray (nlags x nfeats x nchans x regularization params)
        """

        betas = np.reshape(self.coef_, (len(self.lags),
                                        self.n_feats_, self.n_chans_))
        betas = betas[::-1, :]
        return betas


    def predict(self, X):
        """
        Compute output based on fitted coefficients and feature matrix X.
        
        Parameters
        ----------
        X : ndarray
            Matrix of features (can be already lagged or not).
            
        Returns
        -------
        ndarray
            Reconstruction of target with current beta estimates
        """
        
        assert self.fitted, "Fit model first!"

        #betas = self.get_coef()[:]
        betas = self.coef_[:]

        # Check if input has been lagged already, if not, do it:
        X = lag_matrix(X, lag_samples=self.lags, filling=0.)


        # Do it for every alpha
        if not self.blocks:
            pred = X@betas
        else:
            X_blocks = X.reshape(X.shape[0],len(self.times),self.n_feats_)
            X_blocks = list(np.moveaxis(X_blocks, 2, 0))
            pred = 0.0
            for Xk, Bk in zip(X_blocks, self.coef_):
                pred += Xk @ Bk

        return pred  # Shape T x Nchan x Alpha

        
    def score(self, Xtest, ytrue, Xtrain = None, scoring="R2"):
        """Compute a score of the model given true target and estimated target from Xtest.
        
        Parameters
        ----------
        Xtest : ndarray
            Array used to get "yhat" estimate from model
        ytrue : ndarray
            True target
        scoring : str
            Scoring function to be used ("corr", "rankcorr", "rmse", "R2", "ezekiel", 'adj_R2')
            
        Returns
        -------
        scores: ndarray
            Scores computed on whole segment.
        """
        yhat = self.predict(Xtest)

        lags = self.lags
        if scoring == 'corr':
            scores = pearsonr(ytrue, yhat)[0]
            self.scores = scores
            return scores
        elif scoring == 'rmse':
            scores = root_mean_squared_error(ytrue, yhat, multioutput='raw_values')
            self.scores = scores
            return scores
        elif scoring == 'R2':
            scores = r2_score(ytrue, yhat, multioutput='raw_values')
            self.scores = scores
            return scores
        else:
            raise NotImplementedError(
                "Only correlation & RMSE scores are valid for now...")

    def xval_eval(self, X, y, n_splits=5, lagged=False, drop=True, train_full=True, scoring="R2", segment_length=None, fit_mode='direct', verbose=True):
        """
        Standard cross-validation. Scoring. 
        
        Parameters
        ----------
        X : ndarray 
            input of size (T x nfeats)
        y : ndarray 
            output of size (T x nchans)
        n_splits : integer 
            Number of folds, default to 5.
        lagged : bool
            Whether the X matrix has been previously 'lagged', default to False.
        drop : bool
            Whether to drop non valid samples (if False, non valid sample are filled with 0.).
        train_full : bool 
            Train model using all the available data after the end of x-val
        scoring : string
            Scoring method (see scoring()), default to "R2".
        segment_length: integer, float 
            Length of a testing segments (that testing data will be chopped into). If None, use all the available data. Default to None.
        fit_mode : string {'direct' | 'from_cov_xxx'} (default: 'direct')
            Model training mode. 'direct' - fit using all the avaiable data at once (i.e. fit()), 'from_cov_xxx' - fit using all the avaiable data from covariance matrices. The routine will chop data into pieces, compute piece-wise cov matrices and fit the model. 'xxx' portion of the string indicates the lenght of the segments that the data will be chopped into. 
        verbose : bool
        
        Returns
        -------
        scores : ndarray 
            scores across each fold (n_splits x segments x nchans x alpha)
        """

        #if np.ndim(self.alpha) < 1 or len(self.alpha) <= 1:
        #    raise ValueError(
        #        "Supply several alphas to TRF constructor to use this method.")

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
            scores = np.zeros((n_splits, self.n_chans_))

        for kfold, (train, test) in enumerate(kf.split(X)):
            if verbose:
                print("Training/Evaluating fold %d/%d" % (kfold+1, n_splits))

            self.fit(X[train, :], y[train, :])

            if segment_length:  # Chop testing data into smaller pieces

                if (len(test) % segment_length) > 0:  # Crop if there are some odd samples
                    test_crop = test[:-int(len(test) % segment_length)]
                else:
                    test_crop = test[:]

                # Reshape to # segments x segment duration
                test_segments = test_crop.reshape(
                    int(len(test_crop) / segment_length), -1)

                ccs = [self.score(X[test_segments[i], :], y[test_segments[i], :], scoring=scoring, Xtrain = X[train, :]) for i in range(
                    test_segments.shape[0])]  # Evaluate each segment

                scores.append(ccs)
            else:  # Evaluate using the entire testing data
                scores[kfold, :] = self.score(X[test, :], y[test, :], scoring=scoring, Xtrain = X[train, :])

        if segment_length:
            scores = np.asarray(scores)

        if train_full:
            if verbose:
                print("Fitting full model...")
            self.fit(X, y)
        self.scores = scores

        return scores




    def plot_score(self, figax = None, figsize = (5,5), color_type = 'rainbow', 
                   channels = [], title = 'R2 sumary', minscore = -np.inf):
        """
        Plot the score according to the regularization parameters.

        Parameters
        ----------
        figax : tuple
            contains (fig,ax) matplotlib object, if existing, the dimensions should fit. If None, create a new figure.
        figsize : tuple
            (x,y) size of figure
        color_type : str
            cmap to use
        channels : list
            Select a list of channels indices to plot. If empty, all channels are taken into account.
        title : str
            title of figure
        minscore : float
            Only plot channels that have a score above this value

        Returns
        -------
        fig : Figure
            Matplotlib figure object
        ax : Axes
            Matplotlib axis/axes
        """
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
            if np.max(score_chan > minscore):
                ax.plot(self.alpha, score_chan, color = color_map[index_channel], linewidth = 1.5, label = channels[index_channel])
        ax.set_title(title)
        ax.set_xlabel('Alpha')
        ax.set_ylabel('R2')
        ax.set_xticks(self.alpha)
        ax.set_xscale('log')
        ax.plot(self.alpha, np.mean(self.scores[:,:,:],axis = (0,1)), color = 'k', linewidth = 3, linestyle = '--')
        ax.legend()


        return fig, ax

    def plot_kernel(self, figax = None, figsize = False, color_type = 'rainbow', center_line = False,
                    channels = None, features = None, title = 'kernel sumary', minR2 = -np.inf):
        """
        Plot the TRF of the feature requested as a butterfly plot.
        
        Parameters
        ----------
        figax : tuple
            contains (fig,ax) matplotlib object, if existing, the dimensions should fit. If None, create a new figure.
        figsize : tuple
            (x,y) size of figure
        color_type : str
            cmap to use
        center_line : bool
            Whether to plot a line marking the time 0 (i.e. when input and output are in sync)
        channels : list
            Select a list of channels indices to plot. If empty, all channels are taken into account.
        features : list
            Select a list of features indices to plot. If empty, all features are taken into account.
        title : str
            title of figure
        minscore : float
            Only plot channels that have a score above this value

        Returns
        -------
        fig : Figure
            Matplotlib figure object
        ax : Axes
            Matplotlib axis/axes
        """
        
        if not figsize:
            figsize = (15, (self.n_feats_) * 4)
        if figax is None:
            fig,ax = plt.subplots(self.n_feats_,figsize = figsize, sharex = True)
        else:
            fig,ax = figax
        if channels == None:
            channels = np.arange(self.n_chans_)
        if features == None:
            features = np.arange(self.n_feats_)


        color_map = dict()
        for index_channel in range(self.n_chans_):
            color_map[index_channel] = cmaps[color_type](index_channel/self.n_chans_)

        best_alpha = self.get_best_alpha()
        for feat_index in range(self.n_feats_):
            feat = features[feat_index]
            if self.n_feats_ > 1:
                axfeat = ax[feat_index]
            else:
                axfeat = ax
            for chan_index in range(self.n_chans_):
                alpha_index = best_alpha[chan_index]
                chan = channels[chan_index]
                score_chan = np.mean(self.scores[:,chan_index,:],axis = 0)
                if np.max(score_chan > minR2):

                    axfeat.plot(self.times, self.get_coef()[:,feat_index,chan_index, alpha_index], color = color_map[chan_index], linewidth = 1.5, label = chan)
                    axfeat.set_xlabel('Time (s)')
                    axfeat.set_ylabel(feat)
            if center_line:
                axfeat.plot([0,0],[np.min(self.get_coef()[:,feat_index,:, alpha_index]),np.max(self.get_coef()[:,feat_index,:, alpha_index])], color = 'k', linewidth = 1.5, linestyle = '--')
        handles, labels = axfeat.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1.15, 0.8),loc='right')
        if self.n_feats_ > 1:
            ax[0].set_title(title)
        else:
            ax.set_title(title)
        return fig,ax

    def __repr__(self):
        obj = """RRREstimator(
            lam=%s,
            srate=%s,
            tmin=%s,
            tmax=%s,
            n_feats=%s,
            n_chans=%s,
            n_lags=%s,
            features : %s,
            rank: %s,
            blocks: %s
        )
        """ % (self.lam, self.srate, self.tmin, self.tmax,
               self.n_feats_, self.n_chans_, 
               len(self.lags) if self.lags is not None else None, str(self.feat_names_),
               self.rank, self.blocks)
        return obj

