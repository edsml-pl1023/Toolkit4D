import numpy as np
from scipy.signal import find_peaks


def th_entropy_lesf(frag, nbins=65536):
    """_summary_

    Args:
        frag (_type_): _description_
        nbins (_type_): _description_
    """
    # Get the maximum value for the dtype of the image
    max_value = np.iinfo(frag.dtype).max
    # probably apply medium filter before threshold
    hist, _ = np.histogram(frag, bins=nbins, range=(0, max_value))

    vec_E = np.zeros(nbins)
    vec_A = np.zeros(nbins)

    for i in range(nbins):
        vec_E[i] = E(hist, i)
        vec_A[i] = A(hist, i)

    vec_Alog = np.log10(vec_A)
    vec_A_norm = A(hist, nbins - 1) - vec_A
    vec_Alog_norm = np.log10(vec_A_norm)

    vec = (vec_E / vec_A - vec_Alog +
           (E(hist, nbins - 1) - vec_E) / vec_A_norm - vec_Alog_norm)

    # Find the threshold
    ind = np.nanargmin(vec)
    peaks, properties = find_peaks(-vec, prominence=True)
    peaks[np.where(properties['prominences'] > 0.1)[0][0]]

    if len(peaks) > 0 and peaks[0] <= ind:
        ind = peaks[0]

    T = ind - 1
    return T


def E(hist, i):
    hist = hist[:i + 1]
    hist = hist[hist != 0]
    return np.sum(hist * np.log10(hist))


def A(hist, i):
    return np.sum(hist[:i + 1])
