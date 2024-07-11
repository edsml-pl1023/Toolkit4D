from scipy.io import loadmat
from ToolKit4D.pipeline import ToolKitPipeline
import os
import glob
import math


def test_data_match():
    # load mat data
    mat_data = loadmat('./tests/test_data/threshold_values.mat')
    threshold_infos = {} 
    for rock_label in mat_data['grain_value'].dtype.names:
        for day_label in mat_data['grain_value'][rock_label][0][0].dtype.names:
            threshold_value = int(
                (mat_data['grain_value'][rock_label][0][0][day_label]
                 [0][0][0][0][0])
                )
            threshold_infos[rock_label+day_label] = threshold_value

    # load raw images
    test_data_path = './tests/test_data/'
    test_raw_path = os.path.join(test_data_path, '*.raw')
    raw_files = glob.glob(test_raw_path)
    for raw_file in raw_files:
        img_processor = ToolKitPipeline(raw_file)
        img_processor.th_entropy_lesf()
        assert math.isclose(img_processor.entropy_thresh,
                            threshold_infos[img_processor.identifier],
                            abs_tol=5)
