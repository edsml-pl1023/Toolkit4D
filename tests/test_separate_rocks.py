import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.io import loadmat
from ToolKit4D.stages.separate_rocks import imhmin


def test_distance_transform():
    height_map_truth = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1.0, -1.0, -1.0, 0, 0, -1.0, -1.0, 0],
        [0, -1.0, -2.0, -1.0, 0, 0, -1.0, -1.0, 0],
        [0, -1.0, -2.0, -1.0, 0, 0, 0, 0, 0],
        [0, -1.0, -2.0, -1.4142, -1.0, 0, -1.0, -1.0, -1.0],
        [0, -1.0, -1.0, -1.0, -1.0, 0, -1.0, -2.0, -2.0],
        [0, 0, 0, 0, 0, 0, -1.0, -1.0, -1.0],
        [0, 0, -1.0, -1.0, -1.0, 0, 0, 0, 0],
        [0, 0, -1.0, -2.0, -1.0, 0, 0, 0, 0]
        ])

    data = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1],
        [0, 1, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0],
        ]).astype(bool)
    height_map = -distance_transform_edt(data)
    assert np.all(np.isclose(height_map, height_map_truth))


def test_preprocessing():
    mat_data = loadmat('./tests/test_data/distance_map_aux.mat')
    dataStruct = mat_data['dataStruct']
    for i in range(dataStruct.shape[1]):
        data = dataStruct[0, i]['data'].astype(bool)
        height_map_aux_matlab = dataStruct[0, i]['height_map_aux']
        height_map = -distance_transform_edt(data)
        height_map[~data] = -np.inf
        height_unique = np.unique(-height_map)
        height_10percent = height_unique[-2] / 10
        height_map_aux_python = imhmin(height_map, height_10percent)
        assert np.allclose(height_map_aux_python,
                           height_map_aux_matlab, atol=1e-06)
