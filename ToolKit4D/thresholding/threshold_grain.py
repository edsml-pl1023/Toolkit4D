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
    if len(peaks) == 0:
        return ind
    peaks[np.where(properties['prominences'] > 0.1)[0][0]]

    if len(peaks) > 0 and peaks[0] <= ind:
        ind = peaks[0]

    return ind


def th_moments(frag, nbins=65536):
    """_summary_

    Args:
        frag (_type_): _description_
        nbins (int, optional): _description_. Defaults to 65536.
    """

    # Get the maximum value for the dtype of the image
    max_value = np.iinfo(frag.dtype).max
    # probably apply medium filter before threshold
    hist, _ = np.histogram(frag, bins=nbins, range=(0, max_value))
    hist = hist.astype(np.float64)

    Avec = np.zeros(nbins)
    total_sum = A(hist, nbins - 1)

    for i in range(nbins):
        Avec[i] = A(hist, i) / total_sum

    x2 = (B(hist, nbins - 1) * C(hist, nbins - 1) -
          A(hist, nbins - 1) * D(hist, nbins - 1)) / \
         (A(hist, nbins - 1) * C(hist, nbins - 1) - B(hist, nbins - 1)**2)

    x1 = (B(hist, nbins - 1) * D(hist, nbins - 1) -
          C(hist, nbins - 1)**2) / \
         (A(hist, nbins - 1) * C(hist, nbins - 1) - B(hist, nbins - 1)**2)

    x0 = 0.5 - (B(hist, nbins - 1) / A(hist, nbins - 1) + x2 / 2) / \
        np.sqrt(x2**2 - 4 * x1)

    ind = np.nanargmin(abs(Avec-x0))

    return ind


def E(hist, i):
    hist = hist[:i + 1]
    hist = hist[hist != 0]
    return np.sum(hist * np.log10(hist))


def A(hist, i):
    return np.sum(hist[:i + 1])


def B(hist, i):
    # default i is nbins-1=65535
    return np.dot(np.arange(i + 1), hist[:i + 1])


def C(hist, i):
    # default i is nbins-1=65535
    return np.dot((np.arange(i + 1)**2), hist[:i + 1])


def D(hist, i):
    # default i is nbins-1=65535
    return np.dot((np.arange(i + 1)**3), hist[:i + 1])
