"""
iRRR implementation, based on Emily Stephen's python (https://github.com/emilyps14/iRRR_python/tree/master) and Gen Li's Matlab(https://github.com/reagan0323/iRRR) code 
Additional details can be found in Emily P Stephen and Edward F Chang's paper:
Emily P Stephen, Yuanning Li, Sean Metzger, Yulia Oganian, Edward F Chang,
Latent neural dynamics encode temporal context in speech,
Hearing Research,
Volume 437,
2023,
108838,
ISSN 0378-5955,
https://doi.org/10.1016/j.heares.2023.108838.
(https://www.sciencedirect.com/science/article/pii/S0378595523001508)

We changed the format to make it consistent with the SpyEEG library and refactored the code, but the computations are identical to the original library found above.
"""


import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from mne.decoding import BaseEstimator
from ..utils import lag_matrix, lag_span, lag_sparse, mem_check, get_timing, center_weight, count_significant_figures
from ..viz import get_spatial_colors
from scipy import linalg
import mne
from ._methods import _ridge_fit_SVD, _get_covmat, _corr_multifeat, _rmse_multifeat, _objective_value, _soft_threshold
from scipy.linalg import pinv, svd, norm, svdvals

MEM_CAP = 0.9  # Memory cap for the iRRR model (in GB)

class iRRREstimator(BaseEstimator):

    def __init__(self, times=(0.,), tmin=None, tmax=None, srate=1., lambda0 = 0, lambda1 = 1, fit_intercept=False):
        '''
        This class implements the iRRR model for s/M/EEG data.     
        1/(2n)*|Y-1*mu'-sum(X_i*B_i)|^2  + lambda1*sum(w_i*|A_i|_*) (+0.5*lambda0*sum(w_i^2*|B_i|^2_F))  s.t. A_i=B_i

        Parameters
        ----------
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
        lambda1: float (positive)
            Regularization parameter, tuning for nuclear norm
        lambda0: float (positive)
            Default 0. Regularization parameter, tuning for ridge penalty
        param_dict: dict
            Dictionary of parameters for the model. If None, default parameters are used.
                weight: ndarray (K x 1) 
                    Default: np.ones(K)
                    weight vector, theoretically w(i)=(1/n)*max(svd(X{i}))*(sqrt(q)+sqrt(rank(X{i})))
                        heuristically, w(i)=|X_i|_F
                    randomstart: bool or list 
                        Default: False (bool)
                        False or initial condition for B (list)
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

        TODO:   Implement the param_dict
                Consistent typing for randomstart
        
        '''
        # General parameters
        self.tmin = tmin
        self.tmax = tmax
        self.times = times
        self.srate = srate
        self.fitted = False
        self.lags = None
        self.fit_intercept = fit_intercept

        # Fitting parameters
        self.lambdas0 = lambda0
        self.lambda1 = lambda1

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

        # Coefficients of the iRRR
        self.A_ = None
        self.B_ = None
        self.C_ = None
        self.mu_ = None
        self.Theta = None
        self.niter_ = None
        self.rec_Theta_ = None
        self.rec_nonzeros_ = None
        self.rec_primal_ = None
        self.rec_dual_ = None
        self.rec_obj_ = None

    def fill_lags(self):
        """Fill the lags attributes, with number of samples and times in seconds.
        Note
        ----
        Necessary to call this function if one wishes to use iRRR.lags _before_
        :func:`iRRR.fit` is called.

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
        X : list of ndarray nfeats x (T x nlags)
        y : ndarray (T x nchan)
        '''
        self.fill_lags()

        X = np.asarray(X)
        X_list = []
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
        self.n_feats_ = X.shape[1]
        self.n_chans_ = y.shape[1]

        # Assess if feat names corresponds to feat number, fill feat names attributes
        if feat_names:
            err_msg = "Length of feature names does not match number of columns from feature matrix"
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
        for X_f in X.T: X_list.append(lag_matrix(X_f, lag_samples=self.lags, drop_missing=drop, filling=np.nan if drop else 0.))

        # Fill n_samples attribute
        self.n_samples_ = np.sum(self.valid_samples_)

        # Check dimensions and fill n_featlags attribute
        err_msg = "Inconsistent duration between features and EEG, check dimensions"
        p = np.array([Xk.shape[1] for Xk in X_list])
        cumsum_p = np.concatenate([[0],np.cumsum(p)])
        assert(all([Xk.shape[0]==self.n_samples_ for Xk in X_list])),err_msg
        self.n_featlags_ = p
        self.n_featlags_cumsum_ = cumsum_p

        return X_list, y

    def fill_attributes_fit(self, param_dict):
        """Fill the fit parameters attributes, with either default values or parameter dictionary.
        Note
        ----
        Necessary to call this function if one wishes to use iRRR.lags _before_
        :func:`iRRR.fit` is called.
        """
        self.weight = param_dict.get('weight',np.ones((self.n_feats_,1)))
        self.randomstart = param_dict.get('randomstart',False)
        self.varyrho = param_dict.get('varyrho',False)
        self.maxrho = param_dict.get('maxrho',5)
        self.rho = param_dict.get('rho',0.1)
        self.tol = param_dict.get('Tol',1e-3) # stopping rule
        self.Niter = param_dict.get('Niter',500) # Max iterations,

    def initialization(self,X,cX,y, mu, wY1):
    ### initial parameter estimates

        # Initialize B coefficients
        # if randomstart is a list, then it is the initial condition for B, careful with dimensions 
        if isinstance(self.randomstart, list):
            assert(np.all([Bk.shape==(pk,self.n_chans_) for Bk,pk in zip(self.randomstart,self.n_featlags_)]))
            B = [Bk*wk for Bk,wk in zip(self.randomstart,self.weight)]
        # if randomstart is True, B is initialize using random values
        elif self.randomstart:
            B = [randn(pk,q) for pk in self.n_featlags_]
        # if randomstart is False, B is initialize using vanilla OLS
        else:
            B = [pinv(Xk.T @ Xk) @ Xk.T @ wY1 for Xk in X] # OLS

        # Initialize Lagrange parameters for B and concatenate lists
        Theta = [np.zeros((pk,self.n_chans_)) for pk in self.n_featlags_] 
        cB = np.vstack(B) 
        A = B.copy()
        cA = cB.copy()
        cTheta = np.zeros((sum(self.n_featlags_),self.n_chans_))


        _,D_cX,Vh_cX = svd((1/np.sqrt(self.n_samples_))*cX,full_matrices=False)
        if not self.varyrho: # fixed rho
            DeltaMat = Vh_cX.T @ np.diag(1/(D_cX**2+self.lambdas0+self.rho)) @ Vh_cX + \
                (np.eye(sum(self.n_featlags_)) - Vh_cX.T @ Vh_cX)/(self.lambdas0+self.rho)   # inv(1/n*X'X+(lam0+rho)I)
    
        # Compute initial objective values
        obj = [_objective_value(y,X,mu,A,self.lambdas0,self.lambda1),  # full objective function (with penalties) on observed data
            _objective_value(y,X,mu,A,0,0)] # only the least square part on observed data

        return obj, A,cA, cB, Theta, cTheta, DeltaMat
    
    def ADMM(self,obj_init,A,cA,cB,X,cX, meanX, Theta, cTheta, DeltaMat, y,wY1,mu,verbose=False):

        niter = 0
        diff = np.inf
        rec_obj = np.zeros((self.Niter+1,2)) # record objective values
        rec_obj[0,:] = obj_init
        rec_Theta = np.zeros((self.Niter,self.n_feats_)) # record Frobenius norm of Theta
        rec_nonzeros = np.zeros((self.Niter,self.n_feats_)) # record count of nonzero svals
        rec_primal = np.zeros((self.Niter)) # record total primal residual
        rec_dual = np.zeros((self.Niter)) # record total dual residual
        _,D_cX,Vh_cX = svd((1/np.sqrt(self.n_samples_))*cX,full_matrices=False)

        while niter < self.Niter and np.abs(diff)>self.tol:
            if verbose:
                print(f"iRRR iteration {niter+1}/{self.Niter}, diff: {round(diff,count_significant_figures(self.tol) + 1)}/{self.tol}",end="\r")
            niter += 1
            cB_old = cB.copy()

            # estimate concatenated B
            if self.varyrho:
                DeltaMat = Vh_cX.T @ np.diag(1/(D_cX**2+self.lambdas0+self.rho)) @ Vh_cX + \
                    (np.eye(sum(self.n_featlags_)) - Vh_cX.T @ Vh_cX)/(self.lambdas0+self.rho)
            cB = DeltaMat@((1/self.n_samples_)*cX.T@wY1 + self.rho*cA + cTheta)

            # partition cB into components
            B = [cB[cp:nextcp,:] for cp,nextcp in zip(self.n_featlags_cumsum_[:-1],self.n_featlags_cumsum_[1:])]

            # estimate each Ak and update Theta
            for k,(Bk,Thetak) in enumerate(zip(B,Theta)):
                temp = Bk-Thetak/self.rho
                [tempU,tempD,tempVh] = svd(temp,full_matrices=False)
                tempD = _soft_threshold(tempD,self.lambda1/self.rho)
                A[k] = tempU @ np.diag(tempD) @ tempVh
                Theta[k] = Theta[k]+self.rho*(A[k]-Bk)
                rec_nonzeros[niter-1,k] = np.count_nonzero(tempD)
                rec_Theta[niter-1,k] = norm(Theta[k],ord='fro')

            # update cA and cTheta
            for cp,nextcp,Ak,Thetak in zip(self.n_featlags_cumsum_[:-1],self.n_featlags_cumsum_[1:],A,Theta):
                cA[cp:nextcp,:] = Ak
                cTheta[cp:nextcp,:] = Thetak

            # update rho
            if self.varyrho:
                self.rho = min(self.maxrho,1.1*self.rho) # steadily increasing rho

            # check residuals
            primal = norm(cA-cB,ord='fro')**2
            rec_primal[niter-1] = primal
            dual = norm(cB-cB_old,ord='fro')**2
            rec_dual[niter-1] = dual

            # check objective values
            obj = [_objective_value(y,X,mu,A,self.lambdas0,self.lambda1),  # full objective function (with penalties) on observed data
                _objective_value(y,X,mu,A,0,0)] # only the least square part on observed data
            rec_obj[niter,:] = obj

            # stopping rule
            diff = max(primal,self.rho*dual) # primal

        if niter==self.Niter:
            print(f'iRRR does NOT converge after {Niter} iterations!')
        else:
            print(f'iRRR converges after {niter} iterations.')

        # rescale parameter estimate, add back mean
        A = [Ak/w for Ak,w in zip(A,self.weight)]
        B = [Bk/w for Bk,w in zip(B,self.weight)]
        C = np.vstack(A)
        mu = (mu.T - meanX@C).T

        self.niter_ = niter,
        self.rec_Theta_ = rec_Theta[:niter,:],
        self.rec_nonzeros_ = rec_nonzeros[:niter,:],
        self.rec_primal_ = rec_primal[:niter],
        self.rec_dual_ = rec_dual[:niter],
        self.rec_obj_ = rec_obj[:niter+1]

        return C,mu,A,B,Theta

    def fit(self, X, y, lagged=False, drop=True, lambda0 = 0, lambda1 = 1, param_dict = dict(), feat_names = (), verbose = False):
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

        TODO: Optimize Initial fitting for B

        """
        # Preprocess and lag inputs. X is now a list of lagged features matrices
        X, y = self.get_XY(X, y, lagged, drop, feat_names)

        # Adding intercept feature:
        if self.fit_intercept:
            X = np.hstack([np.ones((len(X), 1)), X])

        # Fill fitting parameters attributes
        self.fill_attributes_fit(param_dict)

        # column center Xk's and normalize by the weights
        X, meanX = center_weight(X, self.weight)

        #Stack into usual feature matrix
        cX = np.hstack(X)
        meanX = np.hstack(meanX)

        #Center, mean of Y. We skipped estimation of Y by cutting out the NaN values
        mu = np.mean(y, axis=0, keepdims=True).T #mean estimate of Y
        wY1 = y - mu.T #column centered Y

        obj_init, A, cA, cB, Theta, cTheta, Delta_Mat = self.initialization(X,cX,y,mu,wY1)

        C,mu,A,B,Theta = self.ADMM(obj_init,A,cA, cB,X,cX, meanX, Theta, cTheta, Delta_Mat, y,wY1,mu,verbose)

        return C, mu, A, B, Theta
