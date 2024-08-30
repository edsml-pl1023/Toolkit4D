# Peiyi Leng; edsml-pl1023
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import reconstruction, local_minima
from skimage.measure import label
from skimage.segmentation import watershed


def imcomplement(image):
    """
    Computes the complement of a grayscale image. The complement of each pixel
    is calculated such that the original pixel value and its complement sum to
    1, except for pixels with a value of `-np.inf`, which are mapped to
    `np.inf` in the complement.

    Args:
        image (numpy.ndarray): The input grayscale image as a NumPy array.

    Returns:
        numpy.ndarray: The complemented image, where each pixel is transformed
        according to the rule:
        - If the pixel value is `-np.inf`, it becomes `np.inf`.
        - Otherwise, the pixel value is transformed to `1 - value`.
    """
    # Create the complemented image
    complemented_image = np.where(image == -np.inf, np.inf, 1 - image)

    return complemented_image


def imhmin(img, H):
    """
    Perform H-minima transformation on an image to suppress shallow minima.

    This function applies the H-minima transformation to an image, which
    suppresses all minima in the grayscale image that are shallower than a
    specified depth `H`. The process involves complementing the image,
    performing morphological reconstruction, and then complementing the
    result again.

    Args:
        img (numpy.ndarray): The input grayscale image on which to perform
                             the H-minima transformation.
        H (float): The height threshold. Minima shallower than this value
                   will be suppressed.

    Returns:
        numpy.ndarray: The image after applying the H-minima transformation,
                       with shallow minima removed.
    """
    img = imcomplement(img)
    img = reconstruction(img-H, img)
    img = imcomplement(img)
    return img


def separate_rocks(optimized_mask, suppress_percentage: int = 10,
                   min_obj_size: int = 1000, num_agglomerates: int = 10):
    """
    Separate and filter rock agglomerates in a binary mask using watershed
    segmentation.

    This function processes a binary mask to separate rock agglomerates using
    a distance transform and watershed segmentation. It also filters out small
    agglomerates and returns the largest detected ones based on the specified
    parameters.

    Args:
        optimized_mask (numpy.ndarray): A binary mask where the rock
                                        agglomerates are marked.
        suppress_percentage (int, optional): The percentage used to suppress
                                             shallow minima during the H-minima
                                             transformation. Defaults to 10.
        min_obj_size (int, optional): The minimum size (in pixels) for an
                                      agglomerate to be considered valid.
                                      Defaults to 1000.
        num_agglomerates (int, optional): The maximum number of largest
                                    agglomerates to return. Defaults to 10.

    Returns:
        list[numpy.ndarray]: A list of binary masks, each representing one of
                             the largest detected agglomerates.
    """
    agglomerates = []
    print('\t -- distance map processing ...')
    height_map = -distance_transform_edt(optimized_mask)
    height_map[~optimized_mask] = -np.inf
    height_unique = np.unique(-height_map)
    height_10percent = height_unique[-2] / suppress_percentage
    height_map_aux = imhmin(height_map, height_10percent)

    print('\t -- finding regional minima as marker ...')
    regional_min = local_minima(height_map_aux)
    markers = label(regional_min)
    print('\t -- watershed segmenting ...')
    labels = watershed(height_map_aux, markers)
    unique_labels = np.unique(labels)
    object_lables = np.delete(unique_labels,
                              np.where(unique_labels == 1)[0])
    agglomerates_with_size = []
    print('\t -- removing small agglomerates ...')
    for object_label in object_lables:
        label_mask = (labels == object_label)
        # Filter out small agglomerates
        label_size = np.sum(label_mask)
        if label_size >= min_obj_size:
            agglomerates_with_size.append((label_mask, label_size))
            agglomerates_with_size.sort(key=lambda x: x[1], reverse=True)
            # will only take maximum 10 agglomerates (help in ml)
            top_agglomerates = agglomerates_with_size[:num_agglomerates]
            agglomerates = [mask for mask, _ in top_agglomerates]
    return agglomerates


def binary_search_agglomerates(num_agglomerates, min_obj_size,
                               optimized_rock_mask):
    """
    Perform a binary search to find the optimal suppression percentage that
    yields a desired number of rock agglomerates.

    This function uses a binary search approach to adjust the suppression
    percentage used in the `separate_rocks` function, aiming to obtain a
    specified number of rock agglomerates from a given binary mask. The
    function returns the set of agglomerates that most closely matches the
    desired count.

    Args:
        num_agglomerates (int): The target number of agglomerates to separate
                                from the mask.
        min_obj_size (int): The minimum size (in pixels) for an agglomerate to
                            be considered valid.
        optimized_rock_mask (numpy.ndarray): A binary mask where the rock
                                             agglomerates are marked.

    Returns:
        list[numpy.ndarray]: A list of binary masks, each representing one of
                             the agglomerates that most closely matches the
                             desired count.
    """
    left, right = 1.0, 100.0  # The range for suppress_percentage
    best_agglomerate_masks = None
    best_diff = float('inf')

    while right - left > 0.1:
        mid = (left + right) / 2.0
        suppress_percentage = mid  # Using mid directly as the percentage

        agglomerate_masks = separate_rocks(
            optimized_rock_mask,
            suppress_percentage=suppress_percentage,
            min_obj_size=min_obj_size
        )

        output_length = len(agglomerate_masks)

        if output_length == num_agglomerates:
            return agglomerate_masks
        elif output_length > num_agglomerates:
            right = mid - 0.1  # Adjust right boundary
        else:
            left = mid + 0.1  # Adjust left boundary

        diff = abs(output_length - num_agglomerates)
        if diff < best_diff:
            best_diff = diff
            best_agglomerate_masks = agglomerate_masks

    return best_agglomerate_masks
