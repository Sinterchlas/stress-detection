"""
fft_dft_lib.py
A simple Python library implementing DFT and FFT algorithms from scratch.
"""
import numpy as np

__all__ = ["dft", "fft", "detect_peaks", "adaptive_threshold_peaks", "zscore_peak_detection"]
def zscore_peak_detection(y, lag=5, threshold=3.5, influence=0.5):
    """
    Robust peak detection using z-score method with lag, threshold, and influence.
    Args:
        y (array-like): Input 1D signal.
        lag (int): Window size for moving mean and std.
        threshold (float): Z-score threshold for signaling.
        influence (float): Influence of signals on mean/std (0 = robust, 1 = adaptive).
    Returns:
        signals (np.ndarray): +1 for positive peak, -1 for negative peak, 0 for no peak.
        avgFilter (np.ndarray): Moving average filter.
        stdFilter (np.ndarray): Moving std filter.
    """
    y = np.asarray(y)
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = np.zeros(len(y))
    stdFilter = np.zeros(len(y))
    # Initialize first lag values
    avgFilter[lag-1] = np.mean(y[:lag])
    stdFilter[lag-1] = np.std(y[:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter[i-1]:
            signals[i] = 1 if y[i] > avgFilter[i-1] else -1
            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
        else:
            signals[i] = 0
            filteredY[i] = y[i]
        window = filteredY[max(0, i-lag+1):i+1]
        avgFilter[i] = np.mean(window)
        stdFilter[i] = np.std(window)
    return signals, avgFilter, stdFilter

def dft(x):
    """
    Compute the Discrete Fourier Transform (DFT) of a 1D array x.
    Returns the complex spectrum.
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def fft(x):
    """
    Compute the Fast Fourier Transform (FFT) of a 1D array x using the Cooley-Tukey algorithm.
    Returns the complex spectrum.
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N <= 1:
        return x
    elif N % 2 != 0:
        # fallback to DFT for non-power-of-2 sizes
        return dft(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([
            X_even + factor[:N // 2] * X_odd,
            X_even - factor[:N // 2] * X_odd
        ])

def adaptive_threshold_peaks(signal, window_size=50, factor=1.0, distance=1):
    """
    Detect peaks using adaptive thresholding based on a moving average.
    Args:
        signal (array-like): Input 1D signal.
        window_size (int): Window size for moving average (in samples).
        factor (float): Multiplier for threshold above moving average.
        distance (int): Minimum distance between peaks (in samples).
    Returns:
        peaks (np.ndarray): Indices of detected peaks.
        threshold (np.ndarray): Adaptive threshold array.
    """
    signal = np.asarray(signal)
    mov_avg = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
    mov_std = np.sqrt(np.convolve((signal - mov_avg)**2, np.ones(window_size)/window_size, mode='same'))
    threshold = mov_avg + factor * mov_std
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i] > threshold[i] and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            peaks.append(i)

    if distance > 1 and len(peaks) > 1:
        filtered = [peaks[0]]
        for idx in peaks[1:]:
            if idx - filtered[-1] >= distance:
                filtered.append(idx)
        peaks = filtered
    return np.array(peaks), threshold

def detect_peaks(signal, height=None, distance=None, prominence=None):
    peaks = []
    signal = np.asarray(signal)
    for i in range(1, len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            peaks.append(i)
    # Filter by height
    if height is not None:
        peaks = [i for i in peaks if signal[i] >= height]
    # Filter by distance
    if distance > 1 and len(peaks) > 1:
        filtered = [peaks[0]]
        for i in peaks[1:]:
            if i - filtered[-1] >= distance:
                filtered.append(i)
        peaks = filtered
    # Filter by prominence
    if prominence is not None:
        def get_prom(i):
            left = i
            while left > 0 and signal[left] > signal[left-1]:
                left -= 1
            right = i
            while right < len(signal)-1 and signal[right] > signal[right+1]:
                right += 1
            base = max(signal[left], signal[right])
            return signal[i] - base
        peaks = [i for i in peaks if get_prom(i) >= prominence]
    return np.array(peaks)
