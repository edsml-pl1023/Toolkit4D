# Peiyi Leng; edsml-pl1023
import numpy as np
import tifffile


def tif_read(filename):
    """
    Reads a TIFF file and returns the image data,
    converting it to a boolean array if the data type is `uint8`,
    which is designed specifically for the bool mask stored
    from this package

    Args:
        filename (str): The path to the TIFF image file to be read.

    Returns:
        numpy.ndarray: The image data as a NumPy array. If the
        image data type is `uint8`, it is converted to a boolean array;
        otherwise, the original data type is preserved.
    """
    # Read the TIFF file
    image = tifffile.imread(filename)

    # Check if the image is boolean or uint16 and return it
    if image.dtype == np.uint8:
        return image.astype(bool)
    else:
        return image
