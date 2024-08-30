# Peiyi Leng; edsml-pl1023
import numpy as np
from scipy.signal import find_peaks


def th_entropy_lesf(frag, nbins=65536):
    """
    Calculate the threshold of an image fragment using the maximum
    entropy method.

    This function computes a threshold for the input image fragment (`frag`) by
    analyzing its histogram with the maximum entropy method.
    The process involves:
    - Calculating the histogram of the image fragment.
    - Computing entropy (`vec_E`) and cumulative probability (`vec_A`) vectors.
    - Normalizing these vectors and combining them to find the threshold
      that maximizes the entropy.

    Args:
        frag (numpy.ndarray): The image fragment for which to calculate the
                              threshold. The fragment should be a NumPy array
                              with an integer data type.
        nbins (int, optional): The number of bins to use for the histogram
                               calculation. The default is 65536, suitable for
                               16-bit images.

    Returns:
        int: The index representing the calculated threshold value within the
        histogram bins.
    """
    # Get the maximum value for the dtype of the image
    max_value = np.iinfo(frag.dtype).max

    print('\t -- calculating histogram...')
    # probably apply medium filter before threshold
    hist, _ = np.histogram(frag, bins=nbins, range=(0, max_value))

    vec_E = np.zeros(nbins)
    vec_A = np.zeros(nbins)

    print('\t -- calculating vec ...')
    for i in range(nbins):
        vec_E[i] = E(hist, i)
        vec_A[i] = A(hist, i)

    vec_Alog = np.log10(vec_A)
    vec_A_norm = A(hist, nbins - 1) - vec_A
    vec_Alog_norm = np.log10(vec_A_norm)

    vec = (vec_E / vec_A - vec_Alog +
           (E(hist, nbins - 1) - vec_E) / vec_A_norm - vec_Alog_norm)

    print('\t -- finding min and regional min ...')
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
    """
    Calculate the threshold of an image fragment using the moments method.

    This function computes a threshold for the input image fragment (`frag`)
    by analyzing
    its histogram using the moments algorithm. The moments method involves:
    - Calculating the histogram of the image fragment.
    - Computing moments of the histogram to find parameters (`x0`, `x1`, and
      `x2`) that describe the histogram's distribution.
    - Using these parameters to determine the threshold that best represents
      the image data.

    Args:
        frag (numpy.ndarray): The image fragment for which to calculate the
                              threshold.
                              The fragment should be a NumPy array with an
                              integer data type.
        nbins (int, optional): The number of bins to use for the histogram
                               calculation. Defaults to 65536, suitable for
                               16-bit images.

    Returns:
        int: The index representing the calculated threshold value within the
             histogram bins.
    """

    # Get the maximum value for the dtype of the image
    max_value = np.iinfo(frag.dtype).max
    # probably apply medium filter before threshold
    print('\t -- calculating histogram...')
    hist, _ = np.histogram(frag, bins=nbins, range=(0, max_value))
    hist = hist.astype(np.float64)

    print('\t -- calculating vec ...')
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

    print('\t -- finding min ...')
    ind = np.nanargmin(abs(Avec-x0))

    return ind


def E(hist, i):
    hist = hist[:i + 1]
    hist = hist[hist != 0]
    return np.sum(hist * np.log10(hist))


def A(hist, i):
    return np.sum(hist[:i + 1])


def B(hist, i):
    indices = np.arange(i + 1, dtype=np.int64)
    return np.dot(indices, hist[:i + 1])


def C(hist, i):
    indices = np.arange(i + 1, dtype=np.int64)
    return np.dot(indices**2, hist[:i + 1])


def D(hist, i):
    # default i is nbins-1=65535
    indices = np.arange(i + 1, dtype=np.int64)
    return np.dot(indices**3, hist[:i + 1])
