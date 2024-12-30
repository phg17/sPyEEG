import numpy as np
from scipy.signal.windows import get_window
from scipy.stats import pearsonr
from scipy.fft import fft, ifft
from scipy.signal import fftconvolve, welch
from scipy.signal import csd as welch_csd


def resample_array(array, new_length):
    """
    Resamples an array to a fixed number of points using average pooling.

    Parameters:
        array (np.ndarray): The input array to resample. Can be 1D or multi-channel (e.g., 2D for multi-channel).
        new_length (int): The desired length of the output array.

    Returns:
        np.ndarray: The resampled array.
    """
    if new_length <= 0:
        raise ValueError("new_length must be greater than 0.")

    # Handle multi-channel arrays
    if array.ndim == 1:
        array = array[:, np.newaxis]

    # Original array length
    old_length = array.shape[0]

    if old_length == new_length:
        return array.copy() if array.ndim == 1 else array.copy().squeeze()

    # Compute the resampling ratio
    ratio = old_length / new_length

    # Create new indices
    new_indices = np.linspace(0, old_length, new_length, endpoint=False)

    # Initialize the resampled array
    resampled = np.zeros((new_length, *[array.shape[i] for i in range(1, len(array.shape))]), dtype=array.dtype)

    # Average pooling for the segments
    for i in range(new_length):
        # Determine the range of indices in the original array that contribute to this output index
        start_idx = int(np.floor(i * ratio))
        end_idx = int(np.ceil((i + 1) * ratio))

        # If the range is within bounds, compute the mean
        if end_idx > start_idx:
            resampled[i] = np.mean(array[start_idx:end_idx], axis=0)
        else:
            # Interpolate for fractional indices
            lower_idx = min(start_idx, old_length - 1)
            upper_idx = min(end_idx, old_length - 1)
            weight = (i * ratio - lower_idx)
            resampled[i] = (1 - weight) * array[lower_idx] + weight * array[upper_idx]

    return resampled


class FourierTRFEstimator:
    """
    Estimate the Impulse Response Function (IRF) of a system.

    Supports both time-domain and frequency-domain estimation, with options for
    regularization and negative lags.

    Attributes:
    ----------
    min_lag : int
        Minimum lag to consider (negative for future input).
    max_lag : int
        Maximum lag to consider (positive for past input).
    method : str
        Estimation method ('time' or 'frequency').
    regularization : str or None
        Regularization type ('ridge' or None).
    alpha : float
        Regularization strength (used if regularization is 'ridge').
    irf : ndarray or None
        Estimated impulse response function (set after calling `fit`).

    Methods:
    -------
    fit(x, y):
        Estimate the impulse response function using input-output data.
    predict(x):
        Predict the output signal using the estimated impulse response.
    score(x, y):
        Compute the coefficient of determination (R^2 score) for each output.
    """

    def __init__(self, min_lag, max_lag, method='time', regularization=None, alpha=0.1, eps=1e-6, frequency_smoothing_method='boxcar'):
        """
        Initialize the ImpulseResponseEstimator.

        Parameters:
        ----------
        min_lag : int
            Minimum lag (negative for future input).
        max_lag : int
            Maximum lag (positive for past input).
        method : str, optional
            Estimation method ('time' or 'frequency'), default is 'time'.
        regularization : str or None, optional
            Regularization type ('ridge' or None), default is None.
        alpha : float, optional
            Regularization strength, used if regularization is 'ridge'.
        frequency_smoothing: str
            The frequency-domain smoothing method ('none', 'boxcar', 'welch', 'resample')
        """
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.method = method
        self.regularization = regularization
        self.alpha = alpha
        self.irf = None
        self.eps = eps
        self.smoothing_method = frequency_smoothing_method

    def fit(self, x, y):
        """
        Fit the impulse response function using input-output data.

        Parameters:
        ----------
        x : ndarray of shape (n_samples,)
            Input signal.
        y : ndarray of shape (n_samples, n_outputs)
            Output signal, where each column corresponds to an output feature.

        Returns:
        -------
        irf : ndarray of shape (max_lag - min_lag + 1, n_outputs)
            Estimated impulse response function.
        """
        y = np.atleast_2d(y)  # Ensure y is 2D
        if self.method == 'time':
            self.irf = self._fit_time_domain(x, y)
        elif self.method == 'frequency':
            self.irf = self._fit_frequency_domain(x, y)
        else:
            raise ValueError("Invalid method. Choose 'time' or 'frequency'.")
        return self.irf

    def predict(self, x):
        """
        Predict the output signal using the estimated impulse response.

        Parameters:
        ----------
        x : ndarray of shape (n_samples,)
            Input signal.

        Returns:
        -------
        y_pred : ndarray of shape (n_samples, n_outputs)
            Predicted output signal.
        """
        if self.irf is None:
            raise ValueError("Model not fitted yet. Call fit() before predict().")

        # Convolve the input signal with the impulse response for each output
        n_outputs = self.irf.shape[1]
        lags = np.arange(self.min_lag, self.max_lag + 1)
        y_pred = np.zeros((len(x), n_outputs))

        for i in range(n_outputs):
            y_pred[:, i] = np.sum(
                [np.roll(x, -lag) * self.irf[lag - self.min_lag, i] for lag in lags], axis=0
            )
        return y_pred

    def score(self, x, y):
        """
        Compute the coefficient of determination (R^2 score) for each output.

        Parameters:
        ----------
        x : ndarray of shape (n_samples,)
            Input signal.
        y : ndarray of shape (n_samples, n_outputs)
            Output signal.

        Returns:
        -------
        r2 : ndarray of shape (n_outputs,)
            R^2 scores for each output.
        """
        y = np.atleast_2d(y)
        y_pred = self.predict(x)

        n_outputs = y.shape[1]
        rs = np.zeros(n_outputs)
        for i in range(n_outputs):
            rs[i] = pearsonr(y[:, i], y_pred[:, i])[0]
        return np.mean(rs)

    def _fit_time_domain(self, x, y):
        """
        Estimate the IRF in the time domain using lagged matrix regression.

        Parameters:
        ----------
        x : ndarray of shape (n_samples,)
            Input signal.
        y : ndarray of shape (n_samples, n_outputs)
            Output signal.

        Returns:
        -------
        irf : ndarray of shape (max_lag - min_lag + 1, n_outputs)
            Estimated impulse response function.
        """
        n_samples, n_outputs = y.shape

        # Construct lagged input matrix
        X = self._construct_lagged_matrix(x)
        autocov = X.T @ X
        
        irf = np.zeros((self.max_lag - self.min_lag + 1, n_outputs))
        for i in range(n_outputs):
            if self.regularization == 'ridge':
                # Ridge regression
                reg_matrix = self.alpha * np.eye(X.shape[1]) * np.mean(np.diag(autocov))
                irf[:, i] = np.linalg.inv(autocov + reg_matrix) @ X.T @ y[:, i]
            else:
                # Ordinary least squares
                irf[:, i] = np.linalg.inv(autocov) @ X.T @ y[:, i]
        return irf

    
    def _fit_frequency_domain(self, x, y):
        """
        Estimate the IRF in the frequency domain using FFT.

        Parameters:
        ----------
        x : ndarray of shape (n_samples,)
            Input signal.
        y : ndarray of shape (n_samples, n_outputs)
            Output signal.

        Returns:
        -------
        irf : ndarray of shape (max_lag - min_lag + 1, n_outputs)
            Estimated impulse response function.
        """
        n_samples, n_outputs = y.shape

        # ChatGPT insists on this padding stuff
        # Zero-pad signals to handle negative and positive lags
        total_lags = self.max_lag - self.min_lag + 1
        x_padded = np.pad(x, (0, total_lags), mode='constant')
        y_padded = np.pad(y, ((0, total_lags), (0, 0)), mode='constant')

        if self.smoothing_method == 'none':
            X_fft = fft(x_padded)
            Y_fft = fft(y_padded, axis=0)
    
            psd = np.abs(X_fft) ** 2
            csd = Y_fft * np.conj(X_fft)[:, None]

        if self.smoothing_method == 'boxcar':
            X_fft = fft(x_padded)
            Y_fft = fft(y_padded, axis=0)
    
            psd = np.abs(X_fft) ** 2
            csd = Y_fft * np.conj(X_fft)[:, None]
            
            window = get_window('boxcar', len(X_fft)//total_lags)
            psd = fftconvolve(psd, window, mode='same')
            csd = np.apply_along_axis(lambda m: fftconvolve(m, window, mode='same'), axis=0, arr=csd)

        if self.smoothing_method == 'welch':
            psd = welch(x_padded, nperseg=total_lags, axis=0, return_onesided=False)[1]
            csd = np.apply_along_axis(
                lambda m: welch_csd(x_padded, m, nperseg=total_lags, axis=0, return_onesided=False)[1],
                axis=0,
                arr=y_padded
            )

        if self.smoothing_method == 'resample':
            X_fft = fft(x_padded)
            Y_fft = fft(y_padded, axis=0)
    
            psd = np.abs(X_fft) ** 2
            csd = Y_fft * np.conj(X_fft)[:, None]

            psd = resample_array(psd, total_lags).squeeze()
            csd = resample_array(csd, total_lags)
        
        # Initialize IRF
        irf = np.zeros((total_lags, n_outputs))

        norm_alpha = self.alpha*np.mean(psd)

        # Estimate IRF in the frequency domain for each output
        for i in range(n_outputs):
            # Frequency-domain division (regularized if specified)
            if self.regularization == 'ridge':
                H_fft = csd[:, i] / (psd + norm_alpha + self.eps)
            else:
                H_fft = csd[:, i] / (psd + self.eps)

            # Inverse FFT to get the impulse response
            full_irf = np.real(ifft(H_fft))

            # Align impulse response for the lag range [min_lag, max_lag]
            irf[:, i] = np.roll(full_irf, self.max_lag)[:total_lags][::-1]

        return irf

    def _construct_lagged_matrix(self, x):
        """
        Construct a lagged matrix for the input signal, including negative and positive lags.

        Parameters:
        ----------
        x : ndarray of shape (n_samples,)
            Input signal.

        Returns:
        -------
        X : ndarray of shape (n_samples, max_lag - min_lag + 1)
            Lagged input matrix.
        """
        n_samples = len(x)
        lags = range(self.min_lag, self.max_lag + 1)
        X = np.zeros((n_samples, len(lags)))

        for i, lag in enumerate(lags):
            if lag < 0:
                # Future input: shift x forward
                X[-lag:, i] = x[:n_samples + lag]
            elif lag > 0:
                # Past input: shift x backward
                X[:n_samples - lag, i] = x[lag:]
            else:
                # Zero lag
                X[:, i] = x
        return X


class FourierTRFEstimatorMIMO:
    """
    Estimate the Impulse Response Function (IRF) of a system.

    Now supports multivariate inputs (multiple input features).

    Attributes:
    ----------
    min_lag : int
        Minimum lag to consider (negative for future input).
    max_lag : int
        Maximum lag to consider (positive for past input).
    method : str
        Estimation method ('time' or 'frequency').
    regularization : str or None
        Regularization type ('ridge' or None).
    alpha : float
        Regularization strength (used if regularization is 'ridge').
    irf : ndarray or None
        Estimated impulse response function, shape is (n_lags, n_features, n_outputs).

    Methods:
    -------
    fit(x, y):
        Estimate the impulse response function using input-output data.
    predict(x):
        Predict the output signal using the estimated impulse response.
    score(x, y):
        Compute a correlation-based score for each output.
    """

    def __init__(self, min_lag, max_lag, method='time', regularization=None, alpha=0.1, eps=1e-6):
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.method = method
        self.regularization = regularization
        self.alpha = alpha
        self.irf = None
        self.eps = eps

    def fit(self, x, y):
        """
        Fit the impulse response function using input-output data.
        x : ndarray (n_samples, n_features) or (n_samples,)
        y : ndarray (n_samples, n_outputs) or (n_samples,)
        """
        # Ensure x and y are 2D
        x = np.atleast_2d(x)
        if x.ndim == 1:
            x = x[:, None]  # make it (n_samples, 1) if single input
        y = np.atleast_2d(y)

        if self.method == 'time':
            self.irf = self._fit_time_domain(x, y)
        elif self.method == 'frequency':
            self.irf = self._fit_frequency_domain(x, y)
        else:
            raise ValueError("Invalid method. Choose 'time' or 'frequency'.")

        return self.irf

    def predict(self, x):
        """
        Predict the output signal using the estimated impulse response.

        x : ndarray of shape (n_samples, n_features) or (n_samples,)
        """
        if self.irf is None:
            raise ValueError("Model not fitted yet. Call fit() before predict().")

        x = np.atleast_2d(x)
        if x.ndim == 1:
            x = x[:, None]

        n_samples = x.shape[0]
        n_features = x.shape[1]
        n_lags = self.max_lag - self.min_lag + 1
        n_outputs = self.irf.shape[2]

        lags = np.arange(self.min_lag, self.max_lag + 1)
        y_pred = np.zeros((n_samples, n_outputs))

        # irf shape: (n_lags, n_features, n_outputs)
        # Summation: y_pred(t, o) = sum_f sum_lag x(t-lag,f)*irf(lag,f,o)
        for o in range(n_outputs):
            for f in range(n_features):
                for li, lag in enumerate(lags):
                    shifted = np.roll(x[:, f], -lag)
                    y_pred[:, o] += shifted * self.irf[li, f, o]

        return y_pred

    def score(self, x, y):
        """
        Compute a correlation-based score for each output.
        """
        y = np.atleast_2d(y)
        y_pred = self.predict(x)
        n_outputs = y.shape[1]
        rs = np.zeros(n_outputs)
        for i in range(n_outputs):
            rs[i] = pearsonr(y[:, i], y_pred[:, i])[0]
        return np.mean(rs)

    def _fit_time_domain(self, x, y):
        """
        Estimate the IRF in the time domain using lagged matrix regression.
    
        x : (n_samples, n_features)
        y : (n_samples, n_outputs)
        """
        n_samples, n_features = x.shape
        n_samples_y, n_outputs = y.shape
    
        if n_samples != n_samples_y:
            raise ValueError(f"Number of samples in x ({n_samples}) and y ({n_samples_y}) must match.")
    
        # Construct lagged matrix for all features
        X = self._construct_lagged_matrix(x)  # shape (n_samples, n_lags * n_features)
        n_lags = self.max_lag - self.min_lag + 1
    
        if X.shape[0] != y.shape[0]:
            raise ValueError("The lagged matrix (X) and output matrix (y) must have the same number of samples.")
    
        autocov = X.T @ X  # Shape: (n_lags * n_features, n_lags * n_features)
    
        irf_vec = np.zeros((n_lags * n_features, n_outputs))  # Shape: (n_lags * n_features, n_outputs)
        for i in range(n_outputs):
            if self.regularization == 'ridge':
                reg_matrix = self.alpha * np.eye(X.shape[1]) * np.mean(np.diag(autocov))
                irf_vec[:, i] = np.linalg.solve(autocov + reg_matrix, X.T @ y[:, i])
            else:
                irf_vec[:, i] = np.linalg.solve(autocov, X.T @ y[:, i])
    
        # Reshape IRF vector into (n_lags, n_features, n_outputs)
        irf = irf_vec.reshape(n_features, n_lags, n_outputs)
        return np.swapaxes(irf, 0, 1)

    def _fit_frequency_domain(self, x, y):
        """
        Estimate the IRF in the frequency domain using FFT (MIMO supported).
        """
        n_samples, n_features = x.shape
        n_samples, n_outputs = y.shape

        # what the fuck?
        total_lags = self.max_lag - self.min_lag + 1
        x_padded = np.pad(x, ((0, total_lags), (0, 0)), mode='constant')
        y_padded = np.pad(y, ((0, total_lags), (0, 0)), mode='constant')

        X_fft = fft(x_padded, axis=0)  # Shape: (n_samples + total_lags, n_features)
        Y_fft = fft(y_padded, axis=0)  # Shape: (n_samples + total_lags, n_outputs)
    
        S_xy = X_fft[:, :, None] @ Y_fft[:, None, :].conjugate()
        S_xx = X_fft[:, :, None] @ X_fft[:, None, :].conjugate()

        S_xy = resample_array(S_xy, total_lags*2)
        S_xx = resample_array(S_xx, total_lags*2)

        window = get_window('boxcar', 2)
        S_xy = np.apply_along_axis(lambda m: fftconvolve(m, window, mode='same'), axis=0, arr=S_xy)
        S_xx = np.apply_along_axis(lambda m: fftconvolve(m, window, mode='same'), axis=0, arr=S_xx)
        
        irf = np.zeros((total_lags, n_features, n_outputs))

        reg_matrix = self.alpha * np.eye(S_xx.shape[-1])[None, :, :] * np.diag(np.mean(S_xx, axis=0).real).mean()
        H_fft = np.linalg.inv(S_xx + reg_matrix) @ S_xy

        # Convert transfer function back to time domain
        H_time = np.real(ifft(H_fft, axis=0))  # Time-domain transfer function
    
        # Align IRF to lag range
        for f in range(n_features):
            for o in range(n_outputs):
                irf[:, f, o] = np.roll(H_time[:, f, o], -self.min_lag)[:total_lags]
    
        return irf

    def _construct_lagged_matrix(self, x):
        """
        Construct a lagged matrix for the input signals (multivariate).

        x : (n_samples, n_features)

        Returns:
        X : (n_samples, (max_lag - min_lag + 1)*n_features)
        """
        n_samples, n_features = x.shape
        lags = range(self.min_lag, self.max_lag + 1)
        n_lags = len(lags)

        X_list = []
        # For each feature, create lagged columns and then concatenate
        for f in range(n_features):
            X_f = np.zeros((n_samples, n_lags))
            for i, lag in enumerate(lags):
                if lag < 0:
                    X_f[-lag:, i] = x[:n_samples + lag, f]
                elif lag > 0:
                    X_f[:n_samples - lag, i] = x[lag:, f]
                else:
                    X_f[:, i] = x[:, f]
            X_list.append(X_f)

        # Concatenate all features side by side
        X = np.hstack(X_list)
        return X