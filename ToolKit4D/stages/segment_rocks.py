# Peiyi Leng; edsml-pl1023
import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_opening


def segment_rocks(mask: np.ndarray, d_sample: int = 4, connectivity: int = 2,
                  min_obj_size: int = 1000):
    """
    Returns an optimized mask by downsampling, removing small objects,
    filling holes, and performing morphological operations on the input mask.

    Args:
        mask (np.ndarray): A 3D binary mask, typically derived from
        thresholding operations include `threshold_rock` and `remove_cylinder`.

        d_sample (int, optional): The downsampling factor. The mask will be
        subsampled by this factor along each dimension. Defaults to 4.

        connectivity (int, optional): The connectivity criterion used to
        define connected components.
            - 1: 6-connectivity in 3D.
            - 2: 18-connectivity in 3D (default).
            - 3: 26-connectivity in 3D.

        min_obj_size (int, optional): The minimum size (in voxels) of objects
        to retain in the mask. Objects smaller than this size will be
        removed. Defaults to 1000.

    Returns:
        np.ndarray: A 3D binary mask that has been downsampled,
        cleaned of small objects, had holes filled, and further
        remove small objects through morphological opening.
    """
    # downsample the mask
    downsampled_mask = mask[::d_sample, ::d_sample, ::d_sample]

    print('\t -- filtering out small objects ...')
    # Derive connected objects
    labeled_img = label(downsampled_mask, connectivity=connectivity)
    # Compute the properties of each connected component using regionprops
    props = regionprops(labeled_img)
    # Remove objects smaller than the minimum size
    for prop in props:
        if prop.area < min_obj_size:
            downsampled_mask[labeled_img == prop.label] = False

    print('\t -- filling holes of the image ...')
    # may need to add del mask fill to save ram
    for slice_idx in [0, downsampled_mask.shape[2] - 1]:
        mask_fill = binary_fill_holes(downsampled_mask[:, :, slice_idx])
        downsampled_mask[:, :, slice_idx] = mask_fill
    # Fill holes in the entire 3D image
    downsampled_mask = binary_fill_holes(downsampled_mask)

    print('\t -- finding largest "air" object ...')
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

    print('\t -- do image opening...')
    nhood = get_nhood(connectivity)
    downsampled_mask = binary_opening(downsampled_mask, structure=nhood)

    return downsampled_mask


def get_nhood(connectivity):
    """
    Returns the requested 3D connectivity neighbourhood.

    Args:
        Parameters: selection: 1, 2, or 3
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
