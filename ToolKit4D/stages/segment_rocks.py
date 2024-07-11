import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_opening


def segment_rocks(mask: np.ndarray, d_sample: int = 4, connectivity: int = 2,
                  min_obj_size: int = 2):
    """This function returns an optimized (downsampled /
    remove small object / remove noise) mask

    Args:
        mask (np.ndarray): a binary mask from threshold_rock / remove_cylinder
        d_sample (int, optional): _description_. Defaults to 4.
        connectivity: 1 is 6; 2 is 18; 3 is 26
        min_obj_size: initial: 1000*4*4*4; my downsampled test image (by 8):
                      1000*4*4*4/(8*8*8)=125; my_down 4 agian: 125 / (4*4*4)=2
    """
    # downsample the mask
    downsampled_mask = mask[::d_sample, ::d_sample, ::d_sample]

    # Derive connected objects
    labeled_img = label(downsampled_mask, connectivity=connectivity)
    # Compute the properties of each connected component using regionprops
    props = regionprops(labeled_img)
    # Remove objects smaller than the minimum size
    for prop in props:
        if prop.area < min_obj_size:
            downsampled_mask[labeled_img == prop.label] = False

    for slice_idx in [0, downsampled_mask.shape[2] - 1]:
        mask_fill = binary_fill_holes(downsampled_mask[:, :, slice_idx])
        downsampled_mask[:, :, slice_idx] = mask_fill
    # Fill holes in the entire 3D image
    downsampled_mask = binary_fill_holes(downsampled_mask)

    # Derive connected air objects
    labeled_img_air = label(~downsampled_mask, connectivity=connectivity)
    # Compute the properties of each connected component using regionprops
    props_air = regionprops(labeled_img_air)
    island_sizes = [prop.area for prop in props_air]
    # potential bug: if multiple max; only find the first one (same in Matlab)
    gaps_index = np.argmax(island_sizes)
    island_labels = [prop.label for prop in props_air]
    island_labels.pop(gaps_index)
    # leave only the max size island to false; other to true
    for island_label in island_labels:
        downsampled_mask[labeled_img_air == island_label] = True

    nhood = get_nhood(connectivity)
    downsampled_mask = binary_opening(downsampled_mask, structure=nhood)

    return downsampled_mask


def get_nhood(connectivity):
    """
    Returns the requested 3D connectivity neighbourhood.
    Parameters:
    - selection: 1, 2, or 3
    """
    if connectivity == 1:
        nhood = np.zeros((3, 3, 3), dtype=int)
        nhood[:, :, 0] = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        nhood[:, :, 1] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        nhood[:, :, 2] = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    elif connectivity == 2:
        nhood = np.zeros((3, 3, 3), dtype=int)
        nhood[:, :, 0] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        nhood[:, :, 1] = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        nhood[:, :, 2] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    elif connectivity == 3:
        nhood = np.zeros((3, 3, 3), dtype=int)
        nhood[:, :, 0] = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        nhood[:, :, 1] = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        nhood[:, :, 2] = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    else:
        raise ValueError('Unrecognized selection')

    return nhood
