import numpy as np
from skimage.measure import label, regionprops
from skimage.transform import resize


def agglomerate_extraction(optimized_mask: np.ndarray, raw: np.ndarray,
                           connectivity: int = 2, min_obj_size: int = 2):
    """_summary_

    Args:
        optimized_mask (np.ndarray): _description_
        raw (np.ndarray): _description_
        connectivity (int, optional): _description_. Defaults to 2.
        min_obj_size (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    print('\t -- filtering out small objects ...')
    # Derive connected objects
    labeled_img = label(optimized_mask, connectivity=connectivity)
    # Compute the properties of each connected component using regionprops
    props = regionprops(labeled_img)
    # Remove objects smaller than the minimum size
    for prop in props:
        if prop.area < min_obj_size:
            optimized_mask[labeled_img == prop.label] = False

    print('\t -- upsizing the mask ...')
    # upsizing optimized_mask to raw image dimension
    optimized_mask_int = optimized_mask.astype(int)
    # optimized_mask_full = resize(optimized_mask_int, raw.shape, order=1,
    #                              preserve_range=True, anti_aliasing=True)
    optimized_mask_full = resize(optimized_mask_int, raw.shape, order=0,
                                 preserve_range=True, anti_aliasing=False)
    optimized_mask_full = (optimized_mask_full > 0.5).astype(bool)

    print('\t -- finding bounding box ...')
    # Determine the size of the image
    x, y, z = optimized_mask_full.shape
    top, bottom, left, right, front, back = 0, 0, 0, 0, 0, 0

    # Find the boundaries
    for i in range(x):
        if np.max(optimized_mask_full[i, :, :]) == 1:
            top = i
            break

    for i in range(x-1, -1, -1):
        if np.max(optimized_mask_full[i, :, :]) == 1:
            bottom = i
            break

    for j in range(y):
        if np.max(optimized_mask_full[:, j, :]) == 1:
            left = j
            break

    for j in range(y-1, -1, -1):
        if np.max(optimized_mask_full[:, j, :]) == 1:
            right = j
            break

    for k in range(z):
        if np.max(optimized_mask_full[:, :, k]) == 1:
            front = k
            break

    for k in range(z-1, -1, -1):
        if np.max(optimized_mask_full[:, :, k]) == 1:
            back = k
            break

    # Clip the boundaries to the raw image dimensions
    bottom = min(raw.shape[0], bottom)
    right = min(raw.shape[1], right)
    back = min(raw.shape[2], back)

    print('\t -- extracting frag ...')
    frag_raw = raw[top:bottom+1, left:right+1, front:back+1]
    bw_full = optimized_mask_full[top:bottom+1, left:right+1, front:back+1]
    frag = frag_raw * bw_full.astype(np.uint16)

    return frag
