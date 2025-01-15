"""
Tool to do B2B regression.

This script contains the code for Back-to-Back regression (b2b).
It was originally proposed in:
    King, J. R., Charton, F., Lopez-Paz, D., & Oquab, M. (2020). 
    Back-to-back regression: Disentangling the influence of correlated 
    factors from multivariate observations. NeuroImage, 220, 117028.

Since the original code is not available, this script was written based on the description in the above paper,
and on the description of its implementation in:
    Gwilliams, L., Marantz, A., Poeppel, D., & King, J. R. (2024). 
    Hierarchical dynamic coding coordinates speech comprehension in the brain. bioRxiv, 2024-04.

When to use B2B regression?
B2B regression allows to perform many-to-many regression. It is an improvement over multivariate decoding,
which can only estimate the relation between multiple channels and a single feature (many-to-one). Multivariate
decoding is limited when the features to be decoded are correlated.
It is in these situations that B2B regression is useful, as it can disentangle the influence of multiple correlated features
on multiple channels.

How does B2B regression work?
The principle of B2B is to first perform a regular decoding on half of the data, and then use the second half to perform another 
regression from all true features to each estimated feature. This second regression retrieves the unique relation 
between a true feature and its estimation, knowing all other true features.
Thus, B2B outputs the diagonal of a causal influence matrix, which represents the influence of each feature on all channels. 
The values obtained are beta coefficients. If beta values for a given feature is above 0, it is encoded in the neural signal. 
Note that B2B does not assess significance.

I still don't understand the output of B2B. Can you tell me more ?
In the theorical case where there is no noise in the data, the diagonal of the causal influence matrix S would be a binary matrix.
If a feature i has some influence on the neural data, Sii = 1. If not, Sii = 0. In practice, the noise in the data will make
the estimation fluctuate. The use of Ridge regression allows to reduce the influence of noise, but will produce smaller values in S.
Thus, a practical work-around is to replicate the analysis on multiple subject, and to test whether Sii is significantly above 0.
Note that the values in S are relative to the effect size and to the SNR. 
These values are beta coefficients, and should not be interpreted as explained variance. 
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from ..utils import lag_span, lag_sparse
import mne
from matplotlib import colormaps as cmaps
from sklearn.preprocessing import scale
from ._methods import _ridge_fit_SVD, _get_covmat, _corr_multifeat, _rmse_multifeat, _r2_multifeat, _rankcorr_multifeat, _ezr2_multifeat, _adjr2_multifeat, _b2b
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

class B2BEstimator():
    def __init__(self, tmin, tmax, srate, alphax = np.logspace(-5, 5, 20), alphay = np.logspace(-5, 5, 20)):
        self.srate = srate
        self.tmin = tmin
        self.tmax = tmax
        self.alphax = alphax
        self.alphay = alphay
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

        return X, y

    def xval_eval(self, X, y, n_folds=100, 
                  events = None, ref_index = 0, 
                  events_type = 'single',epoched=False, drop_overlap=True,
                  scalex = False, scaley = False,
                  verbose=True, n_print = 10,
                  n_jobs = -1):
        '''
        to be filled
        '''
        
        if not epoched:
            X,y = self.get_XY(X,y, 
               events = events, ref_index = ref_index, events_type = events_type,
               epoched=False, drop_overlap=drop_overlap, scalex = scalex, scaley = scaley)
            
        S = np.zeros((n_folds, self.epochs_duration, self.n_feats_))
        for i_fold in range(n_folds):
            np.random.seed(i_fold)
            X1, X2, Y1, Y2 = train_test_split(X, y, test_size=0.5)
            if verbose and i_fold % 10 == 0:
                print('Computing fold', i_fold+1, '/', n_folds)
            s = Parallel(n_jobs=n_jobs)(delayed(_b2b)(t,X1,X2,Y1,Y2, self.alphax, self.alphay) for t in range(self.epochs_duration))
            for t in range(self.epochs_duration):
                S[i_fold, t,:] = s[t]

        return S


