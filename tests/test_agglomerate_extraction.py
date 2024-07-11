from scipy.io import loadmat
from ToolKit4D.pipeline import ToolKitPipeline
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import os
import glob
import numpy as np


def test_similarity():
    mat_data = loadmat('./tests/test_data/agglomerate_extraction.mat')

    # load raw images
    test_data_path = './tests/test_data/'
    test_raw_path = os.path.join(test_data_path, '*.raw')
    raw_files = glob.glob(test_raw_path)
    # use these three lists to maintain index info for raw_files
    for raw_file in raw_files:
        img_processor = ToolKitPipeline(raw_file)
        img_processor.agglomerate_extraction()
        frag_matlab = mat_data['data'][img_processor.identifier][0][0]
        frag_python = img_processor.frag
        frag_python_resize = resize(frag_python, frag_matlab.shape,
                                    anti_aliasing=True, preserve_range=True)
        if frag_python_resize.dtype == np.float64:
            frag_python_resize = np.clip(frag_python_resize, 0, 65535)
            frag_python_resize = frag_python_resize.astype(np.uint16)
        similarity_index, _ = ssim(frag_python_resize, frag_matlab,
                                   full=True, data_range=65535)
        assert similarity_index >= 0.7
