# Peiyi Leng; edsml-pl1023
import numpy as np
import tifffile


def grain_mask_write(mask, filename):
    """
    Converts a 3D mask array into a 4D RGBA image and writes it to a file.

    Args:
        mask (numpy.ndarray): A 3D integer array where each voxel value
        corresponds to a specific color:
            - 2: Grain in a rock, will be converted to
            bright blue [0, 0, 255, 255].
            - 1: Matrix in a rock, will be converted to
            bright gray [200, 200, 200, 255].
            - 0: Air in the rock, will be converted to
            transparent [0, 0, 0, 0].
        filename (str): The path where the RGBA image file will be saved.

    """
    rgb_image = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], 4),
                         dtype=np.uint8)  # RGBA

    # Set colors using slicing to avoid dimension mismatch
    rgb_image[mask == 2, :] = [0, 0, 255, 255]  # Bright blue
    rgb_image[mask == 1, :] = [200, 200, 200, 255]  # Bright gray
    rgb_image[mask == 0, :] = [0, 0, 0, 0]  # Transparent

    # Save the image
    tifffile.imwrite(filename, rgb_image)
