import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import reconstruction, local_minima
from skimage.measure import label
from skimage.segmentation import watershed


def imcomplement(image):
    """
    Perform image complement of a grayscale image.
    The resulting image and the original image should sum up to 1 element-wise.
    For -np.inf, it should be np.inf in the complement.

    Parameters:
    image (numpy.ndarray): The input grayscale image.

    Returns:
    numpy.ndarray: The complemented image.
    """
    # Create the complemented image
    complemented_image = np.where(image == -np.inf, np.inf, 1 - image)

    return complemented_image


def imhmin(img, H):
    """_summary_

    Args:
        img (_type_): _description_
        H (_type_): _description_

    Returns:
        _type_: _description_
    """
    img = imcomplement(img)
    img = reconstruction(img-H, img)
    img = imcomplement(img)
    return img


def separate_rocks(optimized_mask, suppress_percentage: int = 10,
                   min_obj_size: int = 5000):
    """_summary_

    Args:
        optimized_mask (_type_): _description_
        suppress_percentage (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    agglomerates = []
    print('--\t distance map processing ...')
    height_map = -distance_transform_edt(optimized_mask)
    height_map[~optimized_mask] = -np.inf
    height_unique = np.unique(-height_map)
    height_10percent = height_unique[-2] / suppress_percentage
    height_map_aux = imhmin(height_map, height_10percent)

    print('--\t finding regional minima as marker ...')
    regional_min = local_minima(height_map_aux)
    markers = label(regional_min)
    print('--\t watershed segmenting ...')
    labels = watershed(height_map_aux, markers)
    unique_labels = np.unique(labels)
    object_lables = np.delete(unique_labels,
                              np.where(unique_labels == 1)[0])
    agglomerates = []
    print('--\t removing small agglomerates ...')
    for object_label in object_lables:
        label_mask = (labels == object_label)
        # Filter out small agglomerates
        if np.sum(label_mask) >= min_obj_size:
            agglomerates.append(label_mask)
    return agglomerates
