from ToolKit4D import dataio
from ToolKit4D import thresholding
from scipy.io import loadmat
import os
import glob
import numpy as np


def test_thresholding_mse():
    """
    Test the mse between thresholding using python threshold_rocks
    and Matlab threshold_rocks (as Ground Truth)
    Mse within a range is acceptable
    """


def test_data_match():
    """
    First, test number of .mat variables are the same as number of raw
    images in test_data
    Second, test they exactly correspond to a same 'image'
    """
    mat_data = loadmat('./tests/test_data/downsampled_raw_image.mat')
    # count the number of meta data in .mat
    meta = sum(1 for key in mat_data.keys() if key.startswith('_'))
    test_data_path = './tests/test_data/'
    test_raw_path = os.path.join(test_data_path, '*.raw')
    raw_files = glob.glob(test_raw_path)
    assert len(mat_data)-meta == len(raw_files)
    for raw_file in raw_files:
        clean_path = os.path.basename(raw_file)
        file_name = os.path.splitext(clean_path)[0]
        fnparts = file_name.split('_')
        mat_variable = '_'.join(fnparts[:3])
        assert mat_variable in list(mat_data.keys())


def test_readraw():
    """
    Test if matlab raw-image variables are the same as python
    read_raw() result
    """
    mat_data = loadmat('./tests/test_data/downsampled_raw_image.mat')

    test_data_path = './tests/test_data/'
    test_raw_path = os.path.join(test_data_path, '*.raw')
    raw_files = glob.glob(test_raw_path)

    for raw_file in raw_files:
        clean_path = os.path.basename(raw_file)
        file_name = os.path.splitext(clean_path)[0]
        fnparts = file_name.split('_')
        mat_variable = '_'.join(fnparts[:3])
        img_matlab = mat_data[mat_variable]
        img_python = dataio.read_raw(
            raw_file,
            [int(fnparts[5]), int(fnparts[6]), int(fnparts[7])],
            fnparts[3])
        print(img_matlab.shape, img_python.shape)

        assert np.array_equal(img_python, img_matlab)
