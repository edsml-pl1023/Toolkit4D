import os
import glob
from ToolKit4D.utils.remove_cylinder import detect_ring
from ToolKit4D.utils import remove_cylinder
from ToolKit4D.dataio import read_raw
from ToolKit4D.thresholding import threshold_rock


def test_detect_ring():
    test_data_path = './tests/test_data/'
    test_raw_path = os.path.join(test_data_path, '*.raw')
    raw_files = glob.glob(test_raw_path)
    for raw_file in raw_files:
        clean_path = os.path.basename(raw_file)
        file_name = os.path.splitext(clean_path)[0]
        fnparts = file_name.split('_')
        img = read_raw(
            raw_file,
            [int(fnparts[5]), int(fnparts[6]), int(fnparts[7])],
            fnparts[3])
        rock_thresh = threshold_rock(raw_image=img)
        img_binary = img >= rock_thresh
        fail = 0
        for slice in range(img_binary.shape[2]):
            _, radius = detect_ring(img_binary[:, :, slice], 99, 150)
            if radius == -1 or radius == -2:
                fail += 1
        fail_rate = fail / float(img_binary.shape[2])
        assert fail_rate <= 0.01


def test_remove_cylinder():
    """
    Test if there is remaining circles in image after
    removing cylinder
    """
    test_data_path = './tests/test_data/'
    test_raw_path = os.path.join(test_data_path, '*.raw')
    raw_files = glob.glob(test_raw_path)
    for raw_file in raw_files:
        clean_path = os.path.basename(raw_file)
        file_name = os.path.splitext(clean_path)[0]
        fnparts = file_name.split('_')
        img = read_raw(
            raw_file,
            [int(fnparts[5]), int(fnparts[6]), int(fnparts[7])],
            fnparts[3])
        rock_thresh = threshold_rock(raw_image=img)
        img_binary = img >= rock_thresh
        img_noColumn = remove_cylinder(img_binary, 99, 1.5)
        fail = 0
        for slice in range(img_noColumn.shape[2]):
            _, radius = detect_ring(img_noColumn[:, :, slice], 99, 150)
            if radius < 0:
                fail += fail
        fail_rate = fail / float(img_noColumn.shape[2])
        assert fail_rate <= 0.001
