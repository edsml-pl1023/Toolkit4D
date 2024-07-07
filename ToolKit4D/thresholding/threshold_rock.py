import os
import numpy as np
from skimage import filters
from ..dataio.read_raw import read_raw


def threshold_rock(raw_file, nbins=256):
    """_summary_

    Args:
        raw_file (str): complete path to a raw file.
            e.g ./tests/test_data/LA2_d0_v1_uint16_unnormalized_254_254_254.raw
        clean_path (str): basename of raw_file
        file_name (str): (raw_file - basename); file_name[0]: without extension
        nbins (int): decide histogram precision; larger takes longer
        time to calculate
    """

    clean_path = os.path.basename(raw_file)
    file_name = os.path.splitext(clean_path)[0]
    fnparts = file_name.split('_')
    im_size = [int(fnparts[5]), int(fnparts[6]), int(fnparts[7])]
    im_type = fnparts[3]

    # probably add more supported types
    if im_type == 'uint16':
        max_value = 65535
    else:
        raise ValueError('Unsupported image type')

    # print('- Loading raw data...')
    raw = read_raw(raw_file, im_size, im_type)
    # print('\t - Done.')

    # probably apply medium filter before threshold
    hist, _ = np.histogram(raw, bins=nbins, range=(0, np.iinfo(raw.dtype).max))

    # first output is thresholding value and the second is thresholded image
    # print("- Finding Otsu's threshold...")
    ret = filters.threshold_otsu(hist=hist)
    # ret = filters.threshold_otsu(raw)
    # print('\t - Rocks threshold calculation is done.')

    return ret*(max_value/(nbins-1))
