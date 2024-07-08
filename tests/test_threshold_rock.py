from ToolKit4D import dataio
from ToolKit4D import thresholding
from scipy.io import loadmat
import os
import glob
import numpy as np


def test_data_match():
    """
    First, test number of .mat threshold info are the same as number of raw
    images in test_data
    Second, test they exactly correspond to a same 'image'; they exactly match
    by two direction assert
    """
    # load mat data
    mat_data = loadmat('./tests/test_data/threshold_values.mat')
    threshold_infos = []
    for rock_label in mat_data['rock_value'].dtype.names:
        for day_label in mat_data['rock_value'][rock_label][0][0].dtype.names:
            threshold_value = int(
                mat_data['rock_value'][rock_label][0][0][day_label]
                )
            # threshold info is a list of tuples, each tuple:
            # (rock label, day label, thres value)
            threshold_infos.append((rock_label, day_label, threshold_value))
    threshold_infos_label = [info[0]+info[1] for info in threshold_infos]

    # load raw images
    test_data_path = './tests/test_data/'
    test_raw_path = os.path.join(test_data_path, '*.raw')
    raw_files = glob.glob(test_raw_path)

    assert len(threshold_infos) == len(raw_files)

    mat_variables = []
    for raw_file in raw_files:
        clean_path = os.path.basename(raw_file)
        file_name = os.path.splitext(clean_path)[0]
        fnparts = file_name.split('_')
        mat_variable = fnparts[0]+fnparts[2]+fnparts[1]
        mat_variables.append(mat_variable)
        assert mat_variable in threshold_infos_label

    for threhold_info_label in threshold_infos_label:
        assert threhold_info_label in mat_variables


def test_thresholding_mse():
    """
    Test the mse between thresholding using python threshold_rocks
    and Matlab threshold_rocks (as Ground Truth)
    Mse within a range is acceptable
    """
    mat_data = loadmat('./tests/test_data/threshold_values.mat')
    threshold_infos = []
    for rock_label in mat_data['rock_value'].dtype.names:
        for day_label in mat_data['rock_value'][rock_label][0][0].dtype.names:
            threshold_value = int(
                mat_data['rock_value'][rock_label][0][0][day_label]
                )
            # threshold info is a list of tuples, each tuple:
            # (rock label, day label, thres value)
            threshold_infos.append((rock_label, day_label, threshold_value))

    # load raw images
    test_data_path = './tests/test_data/'
    test_raw_path = os.path.join(test_data_path, '*.raw')
    raw_files = glob.glob(test_raw_path)

    # use these three lists to maintain index info for raw_files
    mat_variables = []
    imgs_size = []
    imgs_type = []
    for raw_file in raw_files:
        clean_path = os.path.basename(raw_file)
        file_name = os.path.splitext(clean_path)[0]
        fnparts = file_name.split('_')
        mat_variable = fnparts[0]+fnparts[2]+fnparts[1]
        mat_variables.append(mat_variable)
        imgs_size.append([int(fnparts[5]), int(fnparts[6]), int(fnparts[7])])
        imgs_type.append(fnparts[3])

    for threshold_info in threshold_infos:
        raw_file = raw_files[mat_variables.index(
            threshold_info[0]+threshold_info[1]
            )]
        img_size = imgs_size[mat_variables.index(
            threshold_info[0]+threshold_info[1]
            )]
        img_type = imgs_type[mat_variables.index(
            threshold_info[0]+threshold_info[1]
            )]

        rock_thresh_python = thresholding.threshold_rock(raw_file)
        rock_thresh_matlab = threshold_info[2]

        # load data
        img_raw = dataio.read_raw(raw_file, img_size, img_type)

        # apply thresholding
        binary_matlab = img_raw >= rock_thresh_matlab
        binary_python = img_raw >= rock_thresh_python
        difference = binary_matlab ^ binary_python
        relative_error = np.sum(difference) / np.prod(img_size)
        assert relative_error <= 0.05
