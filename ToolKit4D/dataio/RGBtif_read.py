# Peiyi Leng; edsml-pl1023
import numpy as np
import tifffile


def grain_mask_read(filename):
    """
    Reads a 4D RGB image file and converts it into a 3D mask
    array based on specific color mappings.

    Args:
        filename (str): The path to the RGB image file to be read.

    Returns:
        numpy.ndarray: A 3D integer array (mask) where each voxel value
        is determined by the color in the corresponding pixel:
            - 2: For pixels matching the bright blue color [0, 0, 255, 255]
            - 1: For pixels matching the bright gray color [200, 200, 200, 255]
            - 0: For pixels matching the transparent color [0, 0, 0, 0]

    """
    # Read the 4D RGB image
    rgb_image = tifffile.imread(filename)

    # Create an empty mask with the same height, width, and depth
    mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1],
                     rgb_image.shape[2]), dtype=int)

    # Define the color mappings
    bright_blue = [0, 0, 255, 255]
    bright_gray = [200, 200, 200, 255]
    transparent = [0, 0, 0, 0]

    # Convert the RGB values back to the original mask values
    mask[np.all(rgb_image == bright_blue, axis=-1)] = 2
    mask[np.all(rgb_image == bright_gray, axis=-1)] = 1
    mask[np.all(rgb_image == transparent, axis=-1)] = 0

    return mask
