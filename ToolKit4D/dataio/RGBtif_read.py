import numpy as np
import tifffile


def grain_mask_read(filename):
    """_summary_

    Args:
        filename (_type_): _description_

    Returns:
        _type_: _description_
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
