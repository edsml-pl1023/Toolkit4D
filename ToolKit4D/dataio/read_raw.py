# Peiyi Leng; edsml-pl1023
import numpy as np


def read_raw(imname, imsize, imtype, big_endian=False,
             skip=0, transposeflag=False):
    """
    Reads a raw image file and returns it as a NumPy array with
    the specified dimensions and data type.

    Args:
        imname (str): The complete path to the raw image file.
            Example: './tests/test_data/LA2_d20_v1_uint16
            _unnormalised_254_254_253.raw'

        imsize (tuple): A tuple specifying the dimensions of the image
        (e.g., (height, width, depth)).

        imtype (str): The data type of the image, which must be
        one of the following:
            - 'uint8' for unsigned 8-bit integer
            - 'uint16' for unsigned 16-bit integer
            - 'float32' for 32-bit floating point
            - 'float64' for 64-bit floating point

        big_endian (bool, optional): Indicates if the data is stored in
        big-endian format. Defaults to False (little-endian).

        skip (int, optional): The number of bytes to skip at the beginning
        of the file. Defaults to 0.

        transposeflag (bool, optional): Whether to transpose the first
        two dimensions of the image.

    Raises:
        ValueError: If the `imtype` is not supported.
        ValueError: If provided imsize is wrong

    Returns:
        numpy.ndarray: The image data as a NumPy array with
        the specified shape and data type.

    """
    dtype_map = {
        'uint8': 'u1',
        'uint16': 'u2',
        'float32': 'f4',
        'float64': 'f8'
    }

    if imtype not in dtype_map:
        raise ValueError("Unsupported imtype: {}".format(imtype))

    # Set endian type
    endian = '<'
    if big_endian:
        endian = '>'

    # Create the dtype with the correct endianness
    dtype = np.dtype(endian + dtype_map[imtype])

    # Open file and read data
    with open(imname, 'rb') as fid:
        # <these two lines need further improvement; here only skip from
        # entire file but not from every 'pixel'>
        if skip > 0:
            fid.read(skip)  # Skip the specified number of bytes

        # Read the data
        imraw = np.fromfile(fid, dtype=dtype, count=np.prod(imsize))

    # Reshape the array to the specified size
    # Also do the transpose: so the slice of third dimension is orthognal
    # to 'column'; the same as Matlab result
    imraw = imraw.reshape(imsize, order='F')

    # Transpose if required; exchange x and y dimension
    if transposeflag:
        imraw = imraw.transpose((1, 0, 2))

    return imraw
