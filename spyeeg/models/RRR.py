"""
Reduced-Rank Regression (iRRR single-block) using FISTA.
"""

import numpy as np
from sklearn.model_selection import KFold
from mne.decoding import BaseEstimator

from ._methods import fit_iRRR_fista, lag_span, lag_sparse, lag_matrix
from ._methods import _corr_multifeat, _rmse_multifeat, _r2_multifeat, _rankcorr_multifeat, _ezr2_multifeat, _adjr2_multifeat


class RRREstimator(BaseEstimator):
    """
    Single-block iRRR / trace-norm regularized multioutput regression:

        min_B 0.5 ||Y - X B||_F^2 + lam ||B||_*

    Parameters
    ----------
    alpha : list | ndarray
        Regularization strengths (here: lambda for nuclear norm). If list, fit all.
    fit_intercept : bool
        If True, include intercept. Implemented by adding a column of ones to X
        (so intercept is not trace-penalized) OR by centering inside fit_iRRR_fista.
        Default here uses explicit intercept column to avoid penalizing intercept.
    center_X : bool
        If True, center X before fitting (passed to fit_iRRR_fista when fit_intercept=False).
    center_Y : bool
        If True, center Y before fitting (passed to fit_iRRR_fista when fit_intercept=False).
    max_iter, tol, power_iter, fista_restart, step, verbose :
        Passed through to fit_iRRR_fista.
    """

    def __init__(
        self,
        alpha=(0.0,),
        tmin = None, 
        tmax = None,
        srate = 1. ,
        fit_intercept=False,
        center_X=False,
        center_Y=True,
        max_iter=500,
        tol=1e-5,
        power_iter=50,
        fista_restart=True,
        step=None,
        verbose=False,
    ):
        self.alpha = np.asarray(alpha, dtype=float)
        self.tmin = tmin
        self.tmax = tmax
        self.fit_intercept = bool(fit_intercept)
        self.center_X = bool(center_X)
        self.center_Y = bool(center_Y)
        self.srate = srate

        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.power_iter = int(power_iter)
        self.fista_restart = bool(fista_restart)
        self.step = step  # None or float
        self.verbose = bool(verbose)

        self.fitted = False

        # Fitted attributes
        self.coef_ = None          # (n_features(+1), n_channels, n_alpha)
        self.intercept_ = None     # (n_channels, n_alpha) if fit_intercept else None
        self.n_feats_ = None
        self.n_chans_ = None
        self.valid_samples_ = None
        self.scores = None

        # iRRR-specific info (one dict per alpha)
        self.rrr_info_ = None

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

    def fit(self, X, y, drop=True):
        """
        Fit iRRR for each alpha (lambda). Stores coef_ as (d(+1), p, n_alpha).
        """
        X, y = self.get_XY(X, y, drop=drop)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        T, d = X.shape
        _, p = y.shape

        n_alpha = len(self.alpha)
        self.rrr_info_ = [None] * n_alpha

        if self.fit_intercept:
            # Explicit intercept column (NOT trace-penalized if we fit B on augmented X,
            # because trace norm will penalize it. To avoid penalizing intercept, we:
            # 1) center X and Y internally (recommended) OR
            # 2) fit on centered data and keep intercept separately.
            #
            # Here: do the standard trick: center X and Y; fit B on centered; intercept = mean(Y) - mean(X)B
            X_mean = X.mean(axis=0, keepdims=True)
            Y_mean = y.mean(axis=0, keepdims=True)
            Xc = X - X_mean
            Yc = y - Y_mean

            B_all = np.zeros((d, p, n_alpha))
            intercept_all = np.zeros((p, n_alpha))

            for i, lam in enumerate(self.alpha):
                B, info = fit_iRRR_fista(
                    Xc, Yc, lam,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    step=self.step,
                    power_iter=self.power_iter,
                    fista_restart=self.fista_restart,
                    center_X=False,
                    center_Y=False,
                    verbose=self.verbose,
                )
                B_all[:, :, i] = B
                intercept_all[:, i] = (Y_mean - X_mean @ B).ravel()
                info = dict(info)  # defensive copy
                info["X_mean_external"] = X_mean
                info["Y_mean_external"] = Y_mean
                self.rrr_info_[i] = info

            self.coef_ = B_all
            self.intercept_ = intercept_all

        else:
            # Use centering options from fit_iRRR_fista (intercept implicitly handled if center_Y True)
            B_all = np.zeros((d, p, n_alpha))
            for i, lam in enumerate(self.alpha):
                B, info = fit_iRRR_fista(
                    X, y, lam,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    step=self.step,
                    power_iter=self.power_iter,
                    fista_restart=self.fista_restart,
                    center_X=self.center_X,
                    center_Y=self.center_Y,
                    verbose=self.verbose,
                )
                B_all[:, :, i] = B
                self.rrr_info_[i] = info

            self.coef_ = B_all
            self.intercept_ = None

        self.fitted = True
        return self.coef_.copy()

    def predict(self, X):
        """
        Predict Y from X using stored coef_ and intercept_.
        Returns: (T, n_chans, n_alpha)
        """
        assert self.fitted, "Fit model first!"
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D (T, d). Got {X.shape}")

        T = X.shape[0]
        n_alpha = self.coef_.shape[-1]

        pred = np.zeros((T, self.n_chans_, n_alpha), dtype=float)
        for i in range(n_alpha):
            Bi = self.coef_[:, :, i]
            yi = X @ Bi
            if self.fit_intercept:
                yi = yi + self.intercept_[:, i][None, :]
            else:
                # If fit_iRRR_fista centered Y, it stored Y_mean in info; add it back for predictions.
                info = self.rrr_info_[i]
                Y_mean = info.get("Y_mean", None)
                X_mean = info.get("X_mean", None)
                # If centering was enabled inside fit_iRRR_fista, it expects you to apply same centering at predict time:
                if (X_mean is not None) and (np.any(X_mean != 0)):
                    yi = (X - X_mean) @ Bi
                if (Y_mean is not None) and (np.any(Y_mean != 0)):
                    yi = yi + Y_mean
            pred[:, :, i] = yi

        return pred

    def score(self, Xtest, ytrue, Xtrain=None, scoring="R2"):
        """
        Mirror TRFEstimator.score: compute metric per alpha and per channel.
        Returns: (n_chans, n_alpha)
        """
        yhat = self.predict(Xtest)
        reg_len = yhat.shape[-1]

        if scoring == "corr":
            scores = np.stack([_corr_multifeat(yhat[..., a], ytrue, nchans=self.n_chans_) for a in range(reg_len)], axis=-1)
        elif scoring == "rmse":
            scores = np.stack([_rmse_multifeat(yhat[..., a], ytrue) for a in range(reg_len)], axis=-1)
        elif scoring == "R2":
            scores = np.stack([_r2_multifeat(yhat[..., a], ytrue) for a in range(reg_len)], axis=-1)
        elif scoring == "rankcorr":
            scores = np.stack([_rankcorr_multifeat(yhat[..., a], ytrue, nchans=self.n_chans_) for a in range(reg_len)], axis=-1)
        elif scoring == "ezekiel":
            # window_length not meaningful here; keep parity with your function signature
            window_length = Xtest.shape[1]
            scores = np.stack([_ezr2_multifeat(yhat[..., a], ytrue, Xtest, window_length) for a in range(reg_len)], axis=-1)
        elif scoring == "adj_R2":
            # needs Xtrain to compute adjusted R2; use provided Xtrain
            if Xtrain is None:
                raise ValueError("adj_R2 requires Xtrain.")
            # no lags in RRR, so pass lags=None
            scores = np.stack([_adjr2_multifeat(yhat[..., a], ytrue, Xtrain, Xtest, self.alpha[a], lags=None) for a in range(reg_len)], axis=-1)
        else:
            raise NotImplementedError("Valid scoring: corr, rankcorr, rmse, R2, ezekiel, adj_R2")

        self.scores = scores
        return scores

    def xval_eval(
        self,
        X,
        y,
        n_splits=5,
        drop=True,
        train_full=True,
        scoring="R2",
        segment_length=None,
        verbose=True,
    ):
        """
        Cross-validation over time samples (KFold) like TRFEstimator.xval_eval.
        Returns scores with shape:
            - if segment_length is None: (n_splits, n_chans, n_alpha)
            - else: (n_splits, n_segments, n_chans, n_alpha)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if segment_length is not None:
            segment_length = int(segment_length)

        self.n_feats_ = X.shape[1]
        self.n_chans_ = y.shape[1] if y.ndim == 2 else y.shape[2]
        reg_len = len(self.alpha)

        kf = KFold(n_splits=n_splits, shuffle=False)

        if segment_length:
            scores = []
        else:
            scores = np.zeros((n_splits, self.n_chans_, reg_len))

        for kfold, (train, test) in enumerate(kf.split(X)):
            if verbose:
                print(f"Training/Evaluating fold {kfold+1}/{n_splits}")

            # Fit on this fold
            self.fit(X[train, :], y[train, :], drop=drop)

            if segment_length:
                if (len(test) % segment_length) > 0:
                    test_crop = test[:-int(len(test) % segment_length)]
                else:
                    test_crop = test[:]
                test_segments = test_crop.reshape(int(len(test_crop) / segment_length), -1)

                ccs = [
                    self.score(X[test_segments[i], :], y[test_segments[i], :], scoring=scoring, Xtrain=X[train, :])
                    for i in range(test_segments.shape[0])
                ]
                scores.append(ccs)
            else:
                scores[kfold, :] = self.score(X[test, :], y[test, :], scoring=scoring, Xtrain=X[train, :])

        if segment_length:
            scores = np.asarray(scores)

        if train_full:
            if verbose:
                print("Fitting full model...")
            self.fit(X, y, drop=drop)

        self.scores = scores
        return scores

    def get_best_alpha(self):
        """
        Return best alpha index per channel based on mean CV score (like TRFEstimator).
        """
        if self.scores is None:
            raise RuntimeError("Run xval_eval() first to populate self.scores.")
        best_alpha = np.zeros(self.n_chans_, dtype=int)
        if self.scores.ndim == 3:
            # folds x chans x alpha
            mean_scores = np.mean(self.scores, axis=0)  # chans x alpha
            best_alpha = np.argmax(mean_scores, axis=-1)
        elif self.scores.ndim == 4:
            # folds x segments x chans x alpha
            mean_scores = np.mean(self.scores, axis=(0, 1))  # chans x alpha
            best_alpha = np.argmax(mean_scores, axis=-1)
        else:
            raise ValueError(f"Unexpected scores shape: {self.scores.shape}")
        return best_alpha

    def __repr__(self):
        ranks = None
        if self.rrr_info_ is not None:
            try:
                ranks = [int(np.max(info.get("rank", [np.nan]))) if info else None for info in self.rrr_info_]
            except Exception:
                ranks = None

        obj = f"""RRRRstimator(
            alpha={self.alpha},
            fit_intercept={self.fit_intercept},
            center_X={self.center_X},
            center_Y={self.center_Y},
            n_feats={self.n_feats_},
            n_chans={self.n_chans_},
            fitted={self.fitted},
            last_ranks={ranks}
        )"""
        return obj
