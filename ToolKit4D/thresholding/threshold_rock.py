import os
import numpy as np
from skimage import filters
from ..dataio.read_raw import read_raw


def threshold_rock(raw_file=None, raw_image=None, nbins=256):
    """_summary_

    Args:
        raw_file (str): complete path to a raw file.
            e.g ./tests/test_data/LA2_d0_v1_uint16_unnormalized_254_254_254.raw
        clean_path (str): basename of raw_file
        file_name (str): (raw_file - basename); file_name[0]: without extension
        nbins (int): decide histogram precision; larger takes longer
        time to calculate
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
