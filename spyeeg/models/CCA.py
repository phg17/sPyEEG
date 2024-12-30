import time

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA

from ._methods import _inverse_square_root, _pairwise_corr
from ..utils import lag_matrix, lag_span, lag_sparse


class CCAEstimator(BaseEstimator):

    def __init__(self, n_components=None, reg=[0, 0], lags=[[0], [0]], use_pca_dimensionality_reduction=True, n_pca_components=None):

        '''
        Initialises the CCA object with the following parameters:

        Parameters
        ----------
        n_components: int, the number of canonical components to retain
        reg: list of two floats, specifying the regularisation parameter to apply to each dataset
        lags: list of two lists of integers, specifying the lags to apply to each dataset
        pca_threshold: None | float | int indicates the number of PCA components to retain for dimensionality reduction. If None, PCA is not applied.
        If pca_threshold is an integer, it specifies the number of components to keep. If it is a float, it specifies the proportion of variance to keep.

        NOTE: is n_components really useful? There's no overhead to keeping all components in this class... if we were using a truncated svd it would be a different story.
        TODO: the regularisation factor can be made relative, e.g. by pre-multiplying by the mean eigenvalue of the covariance matrix.
        NOTE: I include the PCA dimensionality reduction because this was done in the original CCA paper of ADC. However, consensus seems to be that retaining all components/doing no PCA is best.
        '''

        self.n_components = n_components
        self.reg = reg
        self.lags = lags
        self.n_pca_components = n_pca_components
        self.use_pca_dimensionality_reduction = use_pca_dimensionality_reduction

    
    def _get_covariance_matrices(self, X, Y):

        '''
        Function for computing the auto- and cross-covariance matrices of the input datasets X and Y.

        Parameters
        ----------
        X: numpy array of shape (n_samples, n_features_X)
        Y: numpy array of shape (n_samples, n_features_Y)

        Returns
        -------
        cov_X: numpy array of shape (n_features_X, n_features_X)
        cov_Y: numpy array of shape (n_features_Y, n_features_Y)
        cov_XY: numpy array of shape (n_features_X, n_features_Y)
        '''

        data = [X, Y]

        t0 = time.time()
        lagged_data = [lag_matrix(data[i], self.lags[i], drop_missing = True) for i in range(2)]
        t1 = time.time()
        print('Time taken to make lagged matrices:', t1 - t0)

        t0
        cov_X, cov_Y = [lagged_data[i].T @ lagged_data[i] + self.reg[i] * np.eye(lagged_data[i].shape[1]) for i in range(2)]
        cov_XY = lagged_data[0].T @ lagged_data[1]
        t1 = time.time()
        print('Time taken to compute covariance matrices:', t1 - t0)
        
        return cov_X, cov_Y, cov_XY
    
    
    def fit(self, X, Y):

        '''
        Fits the CCA model to the input datasets X and Y

        Parameters
        ----------
        X: numpy array of shape (n_samples, n_features_X)
        Y: numpy array of shape (n_samples, n_features_Y)

        TODO: add PCA whitening, not too important for univariate features, but may matter for multi-channel EEG unless they have already been z-scored.
        '''

        # First, form the lagged data matrices and calculate the covariance matrices

        cov_X, cov_Y, cov_XY = self._get_covariance_matrices(X, Y)

        if self.use_pca_dimensionality_reduction:

            pca_X, pca_Y = [np.linalg.eigh(cov)[1] for cov in [cov_X, cov_Y]]
            pca_X, pca_Y = [pca[:, -self.n_pca_components:] for pca in [pca_X, pca_Y]]
            
            cov_X, cov_Y = [pca.T @ cov @ pca for cov, pca in zip([cov_X, cov_Y], [pca_X, pca_Y])]
            cov_XY = pca_X.T @ cov_XY @ pca_Y

        # Whiten the cross-covariance matrix

        cov_X_inv_sqrt, cov_Y_inv_sqrt = [_inverse_square_root(cov) for cov in [cov_X, cov_Y]]
        cov_XY_whitened = cov_X_inv_sqrt @ cov_XY @ cov_Y_inv_sqrt

        # Perform SVD on the whitened cross-covariance matrix and find the CCA directions

        U, S, Vt = np.linalg.svd(cov_XY_whitened, full_matrices=False)
        V = Vt.T

        self.coefs_X = cov_X_inv_sqrt @ U
        self.coefs_Y = cov_Y_inv_sqrt @ V

        if self.use_pca_dimensionality_reduction:
            self.coefs_X = pca_X @ self.coefs_X
            self.coefs_Y = pca_Y @ self.coefs_Y

        self.coefs_X = self.coefs_X[:, :self.n_components]
        self.coefs_Y = self.coefs_Y[:, :self.n_components]


    def transform(self, X, Y):

        '''
        Transforms the input datasets X and Y using the CCA model

        Parameters
        ----------
        X: numpy array of shape (n_samples, n_features_X)
        Y: numpy array of shape (n_samples, n_features_Y)

        Returns
        -------
        X_transformed: numpy array of shape (n_samples, n_components_X)
        Y_transformed: numpy array of shape (n_samples, n_components_Y)
        '''
        X_transformed = lag_matrix(X, self.lags[0], drop_missing = True) @ self.coefs_X
        Y_transformed = lag_matrix(Y, self.lags[1], drop_missing = True) @ self.coefs_Y

        return X_transformed, Y_transformed
    

    def fit_transform(self, X, Y):

        self.fit(X, Y)

        return self.transform(X, Y)
    
    
    def predict(self, X, Y):

        raise NotImplementedError('This method is not implemented for the CCA class. Please use the transform method instead.')
    

    def score(self, X, Y):

        '''
        returns the correlation coefficient between each pair of canonical variables]

        Parameters
        ----------
        X: numpy array of shape (n_samples, n_features_X)
        Y: numpy array of shape (n_samples, n_features_Y)

        Returns
        -------
        correlations: numpy array of shape (n_components)
        '''

        X_transformed, Y_transformed = self.transform(X, Y)

        return _pairwise_corr(X_transformed, Y_transformed)
