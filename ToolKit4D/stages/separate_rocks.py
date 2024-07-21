import numpy as np
from skimage.morphology import reconstruction


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
    img = imcomplement(img)
    img = reconstruction(img-H, img)
    img = imcomplement(img)
    return img
