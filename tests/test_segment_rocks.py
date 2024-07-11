from scipy.io import loadmat
from ToolKit4D.pipeline import ToolKitPipeline
import os
import glob
import numpy as np


def test_segment_rocks():
    mat_data = loadmat('./tests/test_data/segment_rock.mat')

    # load raw images
    test_data_path = './tests/test_data/'
    test_raw_path = os.path.join(test_data_path, '*.raw')
    raw_files = glob.glob(test_raw_path)
    # use these three lists to maintain index info for raw_files
    for raw_file in raw_files:
        img_processor = ToolKitPipeline(raw_file)
        img_processor.segment_rocks()
        opMask_matlab = mat_data['data'][img_processor.identifier][0][0]
        opMask_python = img_processor.optimized_mask
        difference = np.logical_xor(opMask_python, opMask_matlab)
        different_pixels = np.sum(difference)
        print('differnet pixels:')
        print(different_pixels)
        total_pixels = np.prod(img_processor.im_size)
        print('image size is:')
        print(img_processor.im_size, total_pixels)
        assert (different_pixels / total_pixels) <= 0.001
