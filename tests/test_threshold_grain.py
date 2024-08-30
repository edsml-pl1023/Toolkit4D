from scipy.io import loadmat
from ToolKit4D.pipeline import ToolKitPipeline
from ToolKit4D.stages import agglomerate_extraction
from ToolKit4D.thresholding import th_entropy_lesf 
import os
import glob
import math
import warnings


def test_entropy():
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for raw_file in raw_files:
            img_processor = ToolKitPipeline(raw_file)
            img_processor.remove_cylinder(ring_rad=99, ring_frac=1.5)
            img_processor.segment_rocks()
            frag_python = agglomerate_extraction(
                img_processor.optimized_rock_mask,
                img_processor.raw)
            grain_thresh = th_entropy_lesf(frag_python)
            img_processor.threshold_grain(method='entropy')
            assert math.isclose(grain_thresh,
                                threshold_infos[img_processor.identifier],
                                abs_tol=5)
