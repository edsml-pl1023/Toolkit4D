import numpy as np
import tifffile


def grain_mask_write(mask, filename):
    """_summary_

    Args:
        mask (_type_): _description_
        filename (_type_): _description_
    """
    rgb_image = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], 4),
                         dtype=np.uint8)  # RGBA

    # Set colors using slicing to avoid dimension mismatch
    rgb_image[mask == 2, :] = [0, 0, 255, 255]  # Bright blue
    rgb_image[mask == 1, :] = [200, 200, 200, 255]  # Bright gray
    rgb_image[mask == 0, :] = [0, 0, 0, 0]  # Transparent

    # Save the image
    tifffile.imwrite(filename, rgb_image)
