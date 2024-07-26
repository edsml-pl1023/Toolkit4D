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

    # Sample more slices for evaluation
    num_samples = 2
    slices = np.linspace(0.3, 0.6, num_samples)
    positions = []
    radii = []

    for i, frac in enumerate(slices):
        print(f'--\t finding {i+1} slice circle ...')
        slice_idx = round(img.shape[2] * frac)
        pos, radius = detect_ring(img[:, :, slice_idx], inner_radius,
                                  outer_radius)
        if radius == -1:
            raise ValueError(f"circle not found at slice: {slice_idx}")
        elif radius == -2:
            raise ValueError(f"multiple circles found at slice: {slice_idx}")

        positions.append(pos)
        radii.append(radius)

    # Calculate gradients and intercepts for all points
    grad_x_list = []
    grad_y_list = []
    incp_x_list = []
    incp_y_list = []

    print('--\t finding slope and incp ...')
    for i in range(num_samples - 1):
        p1 = [round(img.shape[2] * slices[i]), positions[i][0]]
        p2 = [round(img.shape[2] * slices[i+1]), positions[i+1][0]]
        grad_x_list.append(grad(p1, p2))
        incp_x_list.append(incp(p1, p2))

        p1 = [round(img.shape[2] * slices[i]), positions[i][1]]
        p2 = [round(img.shape[2] * slices[i+1]), positions[i+1][1]]
        grad_y_list.append(grad(p1, p2))
        incp_y_list.append(incp(p1, p2))

    # Average gradients and intercepts to reduce error
    grad_x = np.mean(grad_x_list)
    grad_y = np.mean(grad_y_list)
    incp_x = np.mean(incp_x_list)
    incp_y = np.mean(incp_y_list)

    slices = np.arange(img.shape[2])
    ring_centres = np.zeros((img.shape[2], 2))
    ring_centres[:, 0] = slices * grad_x + incp_x
    ring_centres[:, 1] = slices * grad_y + incp_y

    # Loop through masking
    print('--\t removing circles at each slice ...')
    mask_siz = [img.shape[0], img.shape[1]]
    for i in range(img.shape[2]):
        slice_mask = create_mask(mask_siz, ring_centres[i], ring_rad)
        img[:, :, i] = np.logical_and(img[:, :, i], slice_mask)
    return img


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

    # blurred = cv2.GaussianBlur(slice_fill, (9, 9), 2)
    # Apply Canny Edge Detection
    # edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.Canny(slice_fill, 50, 255)

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=2*inner_radius,
        param1=50,  # edge thresh; my edge in range (50,150)
        param2=18,  # higher: more accurate but fewer circles
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
    # Generate grid coordinates
    y, x = np.meshgrid(np.arange(mask_siz[0]),
                       np.arange(mask_siz[1]), indexing='ij')

    # Shift the grid to center the circle
    x = x - centre[0]
    y = y - centre[1]

    # Create the mask
    slice_mask = (x**2 + y**2) < radius**2

    return slice_mask


def grad(p1, p2):
    return (p2[1] - p1[1]) / (p2[0] - p1[0])


def incp(p1, p2):
    return p2[1] - grad(p1, p2) * p2[0]
