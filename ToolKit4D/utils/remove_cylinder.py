import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes


def remove_cylinder(img, ring_rad, ring_frac):
    """_summary_

    Args:
        img (array): A thresholded binary 3D image; note: this image
                     directly after thresholding may be ndtype: bool    
        ring_rad: inner radius of circle of any 3D image
                  slice at z direction (in pixels)
        ring_frac: ratio between the outer radius and inner radius
    """

    inner_radius = ring_rad
    outer_radius = round(ring_rad * ring_frac)

    # choose two reference slice
    slice_top = round(img.shape[2] * 0.6)
    slice_btm = round(img.shape[2] * 0.3)

    top_pos, top_radius = detect_ring(img[:, :, slice_top],
                                      inner_radius, outer_radius)
    btm_pos, btm_radius = detect_ring(img[:, :, slice_btm],
                                      inner_radius, outer_radius)

    if top_radius == -1:
        raise ValueError(f"circle not found at slice: {slice_top}")
    elif top_radius == -2:
        raise ValueError(f"multiple circles found at slice: {slice_top}")
    if btm_radius == -1:
        raise ValueError(f"circle not found at slice: {slice_btm}")
    elif btm_radius == -2:
        raise ValueError(f"multiple circles found at slice: {slice_btm}")

    grad_x = grad([slice_btm, btm_pos[0]], [slice_top, top_pos[0]])
    grad_y = grad([slice_btm, btm_pos[1]], [slice_top, top_pos[1]])
    incp_x = grad([slice_btm, btm_pos[0]], [slice_top, top_pos[0]])
    incp_y = grad([slice_btm, btm_pos[1]], [slice_top, top_pos[1]])

    slices = np.arange(img.shape[2])
    ring_centres = np.zeros((img.shape[2], 2))
    ring_centres[:, 0] = slices * grad_x + incp_x
    ring_centres[:, 1] = slices * grad_y + incp_y

    # Loop through masking
    mask_siz = [img.shape[0], img.shape[1]]
    for i in range(img.shape[2]):
        slice_mask = create_mask(mask_siz, ring_centres[i], ring_rad)
        img[:, :, i] = np.logical_and(img[:, :, i], slice_mask)


def detect_ring(slice, inner_radius, outer_radius):
    """_summary_

    Args:
        slice (_type_): A thresholded binary 2D image
        inner_radius (_type_):
        outer_radius (integer): _description_

    Return:
        pos (1d array with 2 elements): center of the detected
                                           circle at the slice
    """
    slice_fill = binary_fill_holes(slice).astype(np.uint8) * 255
    circles = cv2.HoughCircles(
        slice_fill,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=outer_radius-inner_radius,
        param1=200,  # edge thresh; my edge close to 255
        param2=17,  # higher: more accurate but fewer circles
        minRadius=inner_radius,
        maxRadius=outer_radius
    )
    if circles is None:
        print('No ring detected in slice')
        pos = np.array([-1, -1])
        radius = -1
    elif len(circles[0]) == 1:
        circle = circles[0][0]  # Take the first detected circle
        pos = np.array([circle[0], circle[1]])  # (x, y) position
        radius = circle[2]
        # print(
        #     f'Detected one circle: center (x={circle[0]}, y={circle[1]}) '
        #     f'with radius: {circle[2]}'
        # )
    else:
        print(f'More than one ring detected: {len(circles[0])} rings')
        pos = np.array([-2, -2])
        radius = -2

    return pos, radius


def create_mask(mask_siz, centre, radius):
    """
    Create a circular mask with given size, center, and radius.

    Parameters:
    mask_siz (tuple): Size of the mask (height, width).
    centre (tuple): Coordinates of the center of the circle (x, y).
    radius (int): Radius of the circle.

    Returns:
    numpy.ndarray: Circular mask.
    """
    xx, yy = np.ogrid[:mask_siz[0], :mask_siz[1]]
    xx = xx - centre[0]
    yy = yy - centre[1]
    slice_mask = (xx**2 + yy**2) < radius**2
    return slice_mask


def grad(p1, p2):
    return (p2[1] - p1[1]) / (p2[0] - p1[0])


def incp(p1, p2):
    return (p2[1] - grad(p1, p2) * p2[0])
