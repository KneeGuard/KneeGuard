import scipy.signal as sp
import numpy as np


def gait_seg(mag, alpha=0.2):
    b, a = sp.butter(4, 5, 'low', fs=200)
    mag = sp.filtfilt(b, a, mag)
    mid_thres = alpha * np.min(mag)
    peaks = sp.find_peaks(-mag, prominence=0.8, distance=145)[0]
    peaks = peaks[mag[peaks] < mid_thres]

    return peaks
