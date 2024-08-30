# Peiyi Leng; edsml-pl1023
import os
import numpy as np
from skimage import filters
from ..dataio.read_raw import read_raw


def threshold_rock(raw_file=None, raw_image=None, nbins=256):
    """
    Calculate the Otsu threshold for a raw image using its histogram.

    This function computes a threshold for the input image using Otsu's method,
    which finds the threshold that minimizes the intra-class variance in the
    histogram. The function can either read the image data from a file
    (`raw_file`) or directly from a provided image array (`raw_image`). The
    histogram is calculated with a specified number of bins (`nbins`), which
    affects the precision and computation time of the thresholding.

    Args:
        raw_file (str, optional): The complete path to a raw file containing
            the image data. Example:
            './tests/test_data/LA2_d0_v1_uint16_unnormalized_254_254_254.raw'.
        raw_image (numpy.ndarray, optional): A NumPy array representing the
            image data. Either `raw_file` or `raw_image` must be provided,
            but not both.
        nbins (int, optional): The number of bins to use for the histogram
            calculation. A larger number of bins increases precision but
            also the computation time. Defaults to 256.

    Returns:
        float: The calculated threshold value, scaled according to the maximum
        value of the image data type.

    Raises:
        ValueError: If both `raw_file` and `raw_image` are provided, or if
        neither is provided.
    """
    if raw_file is not None and raw_image is not None:
        raise ValueError('Provide only one: raw_file or raw_image.')

    if raw_image is not None:
        raw = raw_image

    elif raw_file is not None:
        clean_path = os.path.basename(raw_file)
        file_name = os.path.splitext(clean_path)[0]
        fnparts = file_name.split('_')
        im_size = [int(fnparts[5]), int(fnparts[6]), int(fnparts[7])]
        im_type = fnparts[3]
        raw = read_raw(raw_file, im_size, im_type)

    else:
        raise ValueError('Either raw_file or raw_image must be provided.')

    # Get the maximum value for the dtype of the image
    max_value = np.iinfo(raw.dtype).max

    print('\t -- calculating histogram ...')
    # probably apply medium filter before threshold
    hist, _ = np.histogram(raw, bins=nbins, range=(0, max_value))

    print('\t -- finding otsu threshold ...')
    ret = filters.threshold_otsu(hist=hist)

    return ret * (max_value / (nbins - 1))
