"""
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


import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

def b2b(X, Y, alphas = np.logspace(-5, 5, 20), n_folds = 100, normalize = True, n_jobs = -1):
    """
    X: np.array of shape (n_trials, n_features), features. 
    Y: np.array of shape (n_trials, n_channels, n_timepoints), recorded neural signal.
    alphas: np.array, regularization parameters
    n_folds: int, number of folds for cross-validation
    normalize: bool, whether to normalize (zscore) the data or not.
            If True, normalization is done across trials, for each feature, channel and timepoint separately.
            Note that data has to be at least centered for B2B to work.

    returns:
    S: np.array of shape (n_features, n_timepoints), estimated causal influence matrix per timepoint
    """

    _, n_features = X.shape
    _, n_channels, n_timepoints = Y.shape
    print('-- Starting B2B regression --')

    #normalize input data
    if normalize:
        X = zscore(X, axis=0)
        for c in range(n_channels):
            Y[:,c,:] = zscore(Y[:,c,:], axis=0)

    #for each fold
    S = np.zeros((n_features, n_timepoints, n_folds))
    for i_fold in range(n_folds):
        #random half-split of data
        np.random.seed(i_fold)
        X1, X2, Y1, Y2 = train_test_split(X, Y, test_size=0.5)
        print('Computing fold', i_fold+1, '/', n_folds)

        #define b2b function for parallel processing
        def b2b_(t):
            y1 = Y1[:,:,t]
            y2 = Y2[:,:,t]

            #predict each feature Xi from all channels Y (i.e. decoding)
            reg1 = RidgeCV(alphas=alphas, fit_intercept=False, cv = None, scoring = 'neg_mean_squared_error') #with cv = None, efficient Leave-One-Out is used
            reg1.fit(y1, X1)
            G = reg1.coef_.T

            #predict each estimated feature Xi from all true features X
            # reg2 = LinearRegression(fit_intercept=False) #King et al., 2020
            reg2 = RidgeCV(alphas=alphas, fit_intercept=False, cv = None, scoring = 'neg_mean_squared_error') #Gwilliams et al., 2024
            reg2.fit(X2, np.dot(y2, G))
            H = reg2.coef_.T

            #return causal influence matrix
            return H.diagonal()

        #run b2b for each timepoint separately
        s = Parallel(n_jobs=n_jobs)(delayed(b2b_)(t) for t in range(n_timepoints))
        for t in range(n_timepoints):
            S[:,t, i_fold] = s[t]
    
    #average across folds
    S = S.mean(axis=2)

    print('-- B2B completed --')

    return S