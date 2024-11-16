"""
Tool to do ERP regression.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from ..utils import lag_span, lag_sparse, get_timing
import mne
from matplotlib import colormaps as cmaps
from sklearn.preprocessing import scale
from ._methods import _ridge_fit_SVD, _get_covmat, _corr_multifeat, _rmse_multifeat, _r2_multifeat, _rankcorr_multifeat, _ezr2_multifeat, _adjr2_multifeat
from sklearn.model_selection import KFold

class ERPEstimator():
    def __init__(self, tmin, tmax, srate, alpha = [0.]):
        self.srate = srate
        self.tmin = tmin
        self.tmax = tmax
        self.alpha = alpha
        self.window = lag_span(tmin, tmax, srate)
        self.times = self.window/srate
        self.epochs_duration = len(self.window)
        self.events = None
        self.indices = None
        self.Xepochs_ = None
        self.Yepochs_ = None
        self.n_chans_ = None
        self.n_epochs_ = None
        self.n_feats_ = None

    def into_epochs(self, X, y, 
                    events = None, ref_index = 0, 
                    events_type= 'single',  
                    drop_overlap = False):
        '''
        Preprocess X and y before fitting. Cutting X and y according to an event reference. 
        If no reference is provided, the program will use the feature at position ref_index instead.
        Parameters
        ----------
        X : ndarray (T x nfeat)
        y : ndarray (T x nchan)
        events : ndarray (n_epochs)
        ref_index : int 
                If no event array is provided, the feature at position ref_index is used instead.       
        events_type : str ('single', 'mean', 'max')
                whether to use the value of features aligned with the events (single), 
                the (mean/max) over the time window,
        drop_overlap : bool
            Default: True.
            Whether to drop non valid samples 
        Returns
        -------
        Features preprocessed for fitting the model.
        Xepochs : ndarray (n_epochs x n_feats)
        yepochs : ndarray (n_epochs x n_times x n_chan)
        '''

        if events is None:
            events = np.where(X[:,ref_index] != 0)[0]
            
        indices = np.arange(int(self.tmin*self.srate), int(self.tmax*self.srate)) + events[:, None]

        # Mask for valid epochs within bounds
        valid_mask = (indices >= 0) & (indices < y.shape[0])
        valid_epochs = np.all(valid_mask, axis=1)
        indices = indices[valid_epochs]
        events = events[valid_epochs]

        # Mask overlapping epochs
        if drop_overlap:
            starts = indices[:, 0]
            sorted_indices = np.argsort(starts)
            indices = indices[sorted_indices]
            events = events[sorted_indices]
    
            # Remove overlapping epochs
            non_overlapping = [0]  # Always keep the first epoch
            for i in range(1, len(indices)):
                if indices[i, 0] > indices[non_overlapping[-1], -1]:
                    non_overlapping.append(i)
    
            indices = indices[non_overlapping]
            events = events[non_overlapping]
            
        self.n_epochs_ = len(indices)
        
        # Extract eeg epochs
        yepochs = np.zeros((self.n_epochs_, self.epochs_duration, self.n_chans_), dtype=y.dtype)
        yepochs[:, :, :] = y[indices]

        # Extract features epochs
        Xepochs = np.zeros((self.n_epochs_, self.n_feats_), dtype=y.dtype)
        if events_type == 'single':
            Xepochs[:,:] = X[events,:]
        elif events_type == 'mean':
            Xepochs[:,:] = X[indices].mean(1)
        elif events_type == 'max':
            Xepochs[:,:] = X[indices].max(1)

        self.events = events
        self.indices = indices
            
        return Xepochs, yepochs
            

    def get_XY(self, X, y, 
               events = None, ref_index = 0, events_type = 'single',
               epoched=False, drop_overlap=True, scalex = True, scaley = True):
        '''
        Preprocess X and y before fitting
        Parameters
        ----------
        X : ndarray (T x nfeat) or ndarray (nepochs x nfeat)
        y : ndarray (T x nchan) or ndarray (nepochs x epochs_window x nchan))
        events : ndarray (n_epochs) 
            array of center of epochs 
        ref_index : int
            if events is None, take the regressor at position ref_index as the center of events
        events_type : str ('single', 'mean', 'max')
                whether to use the value of features aligned with the events (single), 
                the (mean/max) over the time window,
        epoched : bool
            Default: False.
            Whether X and y are epoched
        scalex : bool
            Whether to scale the features
        scaley : bool
            Whether to scale the eeg data
        drop : bool
            Default: True.
            Whether to drop non valid samples
        Returns
        -------
        Features preprocessed for fitting the model.
        X : ndarray (n_epochs x nfeats)
        y : ndarray (n_epochs x epochs_duration * nchan)
        '''

        X = np.asarray(X)
        y = np.asarray(y)

        if not epoched:
            self.n_chans_ = y.shape[1]
            self.n_feats_ = X.shape[1]
            X, y = self.into_epochs(X,y, events, ref_index, drop_overlap = drop_overlap, events_type = events_type)
        else:
            self.n_chans_ = y.shape[2]
            self.n_feats_ = X.shape[1]

        self.Xepochs_ = X.copy()
        self.yepochs_ = y.copy()

        if scalex:
            X = scale(X)
        if scaley:
            for i_chan in range(self.n_chans_):
                y[:,:,i_chan] = scale(y[:,:,i_chan])

        y = np.reshape(y, (self.n_epochs_, self.epochs_duration*self.n_chans_))

        return X, y
        

    def fit(self, X, y,
            events = None, ref_index = 0, events_type = 'single', 
            epoched=False, drop_overlap=True,
            scalex = False, scaley = False):
        """Fit the ERP regression model.
        Parameters
        ----------
        X : ndarray (T x nfeat) or ndarray (nepochs x nfeat)
        y : ndarray (T x nchan) or ndarray (nepochs x epochs_window x nchan))
        events : ndarray (n_epochs) 
            array of center of epochs 
        ref_index : int
            if events is None, take the regressor at position ref_index as the center of events
        events_type : str ('single', 'mean', 'max')
                whether to use the value of features aligned with the events (single), 
                the (mean/max) over the time window,
        epoched : bool
            Default: False.
            Whether X and y are epoched
        scalex : bool
            Whether to scale the features
        scaley : bool
            Whether to scale the eeg data
        drop_overlap : bool
            Default: True.
            Whether to drop non valid samples
        Returns
        -------
        coef_ : ndarray (alphas x nlags x nfeats)
        intercept_ : ndarray (nfeats x 1)
        """

        # Preprocess and lag inputs
        if not epoched:
            X, y = self.get_XY(X, y, 
                   events = events, ref_index = ref_index, events_type = events_type,
                   epoched=epoched, drop_overlap=drop_overlap, scalex = scalex, scaley = scaley)

        # Regress with Ridge to obtain coef for the input alpha
        self.coef_ = _ridge_fit_SVD(X, y, self.alpha, n_feat=self.n_feats_)

        self.fitted = True

        return self

    def get_coef(self):
        '''
        Format and return coefficients.

        Returns
        -------
        coef_ : ndarray (nlags x nfeats x nchans x regularization params)
        '''
        if np.ndim(self.alpha) == 0:
            betas = np.reshape(self.coef_, (self.n_feats_, self.epochs_duration ,self.n_chans_))
        else:
            betas = np.reshape(self.coef_, (self.n_feats_, self.epochs_duration,self.n_chans_, len(self.alpha)))
        return betas

    def predict(self, X):
        """Compute output based on fitted coefficients and feature matrix X.
        Parameters
        ----------
        X : ndarray
            Matrix of features.
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

        betas = self.coef_[:]
        pred = np.stack([X.dot(betas[..., i])
                         for i in range(betas.shape[-1])], axis=-1)
        if np.ndim(self.alpha) == 0:
            pred = np.reshape(pred, (X.shape[0], self.epochs_duration,self.n_chans_))
        else:
            pred = np.reshape(pred, (X.shape[0], self.epochs_duration,self.n_chans_, len(self.alpha)))

        return pred  # Shape Nepochs x Tepochs x Nchan x Nalpha

    def score(self, Xtest, ytrue, Xtrain = None, scoring= "R2"):
        """Compute a score of the model given true target and estimated target from Xtest.
        Parameters
        ----------
        Xtest : ndarray
            Array used to get "yhat" estimate from model
        ytrue : ndarray
            True target
        scoring : str (or func in future?)
            Scoring function to be used ("rmse", "R2")
        Returns
        -------
        float
            Score value computed on whole segment.
        """
        yhat = self.predict(Xtest)
        ytrue = ytrue.reshape((ytrue.shape[0],self.epochs_duration, self.n_chans_))
        reg_len = len(self.alpha)
        alpha = self.alpha
        if scoring == 'rmse':
            scores = np.stack([_rmse_multifeat(yhat[..., a], ytrue) for a in range(reg_len)], axis=-1)
            self.scores = scores
            return scores
        elif scoring == 'R2':
            scores = np.stack([_r2_multifeat(yhat[..., a], ytrue) for a in range(reg_len)], axis=-1)
            self.scores = scores
            return scores
        else:
            raise NotImplementedError(
                "Only R2 & RMSE scores are valid for now...")

    def xval_eval(self, X, y, n_splits=5, 
                  events = None, ref_index = 0, events_type = 'single',epoched=False, 
                  drop_overlap=True, train_full=True, 
                  scalex = False, scaley = False,
                  scoring="R2", verbose=True):
        '''
        to be filled
        '''
        if not epoched:
            X,y = self.get_XY(X,y, 
               events = events, ref_index = ref_index, events_type = events_type,
               epoched=False, drop_overlap=drop_overlap, scalex = scalex, scaley = scaley)
        reg_len = len(self.alpha)
        
        kf = KFold(n_splits=n_splits)
        scores = np.zeros((n_splits, self.epochs_duration ,self.n_chans_, reg_len))
        for kfold, (train, test) in enumerate(kf.split(X)):
            if verbose:
                print("Training/Evaluating fold %d/%d" % (kfold+1, n_splits))
                self.fit(X[train, :], y[train, :], epoched = True)
                scores[kfold, :] = self.score(X[test, :], y[test, :], scoring=scoring, Xtrain = X[train, :])

        if train_full:
            if verbose:
                print("Fitting full model...")
                self.fit(X, y, epoched = True)
        self.scores = scores

        return scores




