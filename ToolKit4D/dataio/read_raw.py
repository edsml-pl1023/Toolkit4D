import numpy as np


def read_raw(imname, imsize, imtype, big_endian=False,
             skip=0, transposeflag=False):
    """_summary_

    Args:
        imname (_type_): complete path to a raw image; e.g.
            './tests/test_data/LA2_d20_v1_uint16_unnormalised_254_254_253.raw'
        imsize (_type_): _description_
        imtype (_type_): _description_
        big_endian (bool, optional): _description_. Defaults to False.
        skip (int, optional): _description_. Defaults to 0.
        transposeflag (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
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
        # entire file but not from envery 'pixel' also and big endian
        # means only use when big endian>
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
