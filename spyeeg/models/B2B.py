"""
Back to back regression, as described in:
King, J. R., Charton, F., Lopez-Paz, D., & Oquab, M. (2020). 
Back-to-back regression: Disentangling the influence of correlated factors from multivariate observations. 
NeuroImage, 220, 117028.
"""

import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from mne.decoding import BaseEstimator
from ..utils import lag_matrix, lag_span, lag_sparse, mem_check, get_timing
from ..viz import get_spatial_colors
from scipy import linalg
import mne
from ._methods import _ridge_fit_SVD, _get_covmat, _corr_multifeat, _rmse_multifeat, _r2_multifeat
from matplotlib import colormaps as cmaps

# Memory cap (i.e. max usage).
# By default set to 90% to prevent bricking machines in corner cases...
MEM_CAP = 0.9


class B2B(BaseEstimator):

    def __init__(self, times=(0.,), tmin=None, tmax=None, srate=1., alphax=[0.], alphay=[0.]):
        '''
        This class implements the back to back regression model for s/M/EEG data.
        times : mismatch a -> b, where a - dependent, b - predicted
            Negative timelags indicate a lagging behind b
            Positive timelags indicate b lagging behind a
        tmin : float
            Default: None
            Minimum time lag (in seconds). Can be negative to check for lags in the past (~null model).
        tmax : float
            Default: None
            Maximum time lag (in seconds). Can be large to check for lags in the future (~null model).
        srate : float
            Default: 1.
            Sampling rate of the data.
        alphax : list
            Default: [0.]
            Regularization parameter(s) for the decoding model. If a list is provided, the model will be fitted for each alphax.
        alphay : list
            Default: [0.]
            Regularization parameter(s) for the source estimation model. If a list is provided, the model will be fitted for each alphay.

        TODO:
            - Implement a method to compute alpha from the data (e.g. nested cross-validation) directly in the function
            - Give the possibility to compute alphas for each feature separately and fit them.
        
        '''

        self.tmin = tmin
        self.tmax = tmax
        self.times = times
        self.srate = srate
        self.alphax = alphax
        self.alphay = alphay
        self.alpha_feat = alpha_feat
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
        # Scores when computed
        self.scores = None

    def fill_lags(self):
        """Fill the lags attributes, with number of samples and times in seconds.
        Note
        ----
        Necessary to call this function if one wishes to use trf.lags _before_
        :func:`trf.fit` is called.

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
        if estimated_mem_usage/1024.**3 > MEM_CAP*mem_check():
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
        """Fit the B2B model.
        First we map y -> X. Note the convention of timelags implies y is not 
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
        self.coef_ = _ridge_fit_SVD(X, y, self.alpha, alpha_feat = self.alpha_feat, n_feat=self.n_feats_)

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
        elif self.alpha_feat:
            betas = np.reshape(self.coef_, (len(self.lags),
                                            self.n_feats_, self.n_chans_, np.power(len(self.alpha), self.n_feats_)))
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

    def fit_direct_cov(self, XXcov=None, XYcov=None, clear_after=True):

        self.XtX_ = XXcov
        self.XtY_ = XYcov

        self.fit_intercept = False
        self.intercept_ = None
        self.coef_ = _ridge_fit_SVD(
            self.XtX_, self.XtY_, self.alpha, from_cov=True, alpha_feat = self.alpha_feat, n_feat = self.n_feats_)
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
            self.XtX_, self.XtY_, self.alpha, from_cov=True,
            alpha_feat = self.alpha_feat, n_feat=self.n_feats_)
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
        pred = np.stack([X.dot(betas[..., i])
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
            Scoring function to be used ("corr", "rmse", "R2")
        Returns
        -------
        float
            Score value computed on whole segment.
        """
        yhat = self.predict(Xtest)
        if self.alpha_feat:
            reg_len = np.power(len(self.alpha), self.n_feats_)
        else:
            reg_len = len(self.alpha)
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
        else:
            raise NotImplementedError(
                "Only correlation & RMSE scores are valid for now...")

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

        if self.alpha_feat:
            reg_len = np.power(len(self.alpha), self.n_feats_)
        else:
            reg_len = len(self.alpha)

        kf = KFold(n_splits=n_splits)
        if segment_length:
            scores = []
        else:
            scores = np.zeros((n_splits, self.n_chans_, reg_len))

        for kfold, (train, test) in enumerate(kf.split(X)):
            if verbose:
                print("Training/Evaluating fold %d/%d" % (kfold+1, n_splits))

            # Fit using trick with adding covariance matrices -> saves RAM
            if fit_mode.find('from_cov') > -1:
                # Format 'from_cov_xxx' -> xxx - duration of a single part.
                # The main idea is to chunk the data into bite-sized parts that will fit in the RAM
                # Careful with the chunking of data
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

                ccs = [self.score(X[test_segments[i], :], y[test_segments[i], :], scoring=scoring) for i in range(
                    test_segments.shape[0])]  # Evaluate each segment

                scores.append(ccs)
            else:  # Evaluate using the entire testing data
                scores[kfold, :] = self.score(X[test, :], y[test, :], scoring=scoring)

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
        self.scores = scores

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

    def get_best_alpha(self):
        best_alpha = np.zeros(self.n_chans_)
        for chan in range(self.n_chans_):
            if len(self.scores.shape) == 3:
                best_alpha[chan] = np.argmax(np.mean(self.scores[:,chan,:],axis=0))
            else:
                best_alpha[chan] = np.argmax(self.scores[:,chan,:],axis=0)
        return best_alpha.astype(int)

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

        return fig,ax

    def plot_score(self, figax = None, figsize = (5,5), color_type = 'jet', 
                   channels = None, title = 'R2 sumary', minR2 = -np.inf):
        if figax == None:
            fig,ax = plt.subplots(figsize = figsize)
        else:
            fig,ax = figax
        if channels == None:
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

    def plot_kernel(self, figax = None, figsize = (15,15), color_type = 'jet', center_line = True,
                    channels = None, features = None, title = 'kernel sumary', minR2 = -np.inf):
        """Plot the TRF of the feature requested as a *butterfly* plot"""
        if figax == None:
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
            for chan_index in range(self.n_chans_):
                alpha_index = best_alpha[chan_index]
                chan = channels[chan_index]
                score_chan = np.mean(self.scores[:,chan_index,:],axis = 0)
                if np.max(score_chan > minR2):
                    ax[feat_index].plot(self.times, self.get_coef()[:,feat_index,chan_index, alpha_index], color = color_map[chan_index], linewidth = 1.5, label = chan)
                    ax[feat_index].set_xlabel('Time (s)')
                    ax[feat_index].set_ylabel(feat)
            if center_line:
                ax[feat_index].plot([0,0],[np.min(self.get_coef()[:,feat_index,:, alpha_index]),np.max(self.get_coef()[:,feat_index,:, alpha_index])], color = 'k', linewidth = 1.5, linestyle = '--')
        handles, labels = ax[1].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1.15, 0.8),loc='right')
        ax[0].set_title(title)
        return fig,ax


