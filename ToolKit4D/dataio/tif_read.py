import numpy as np
import tifffile


def tif_read(filename):
    # Read the TIFF file
    image = tifffile.imread(filename)

    # Check if the image is boolean or uint16 and return it
    if image.dtype == np.uint8:
        return image.astype(bool)
    else:
        return image
