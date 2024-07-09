import numpy as np


def segment_rocks(mask: np.ndarray, d_sample: int = 4, connectivity: int = 18,
                  min_obj_size: int = 1000):
    """This function returns an optimized (downsampled /
    remove small object / remove noise) mask

    Args:
        mask (np.ndarray): a binary mask from threshold_rock / remove_cylinder
        d_sample (int, optional): _description_. Defaults to 4.
    """
    # downsample the mask

    pass
