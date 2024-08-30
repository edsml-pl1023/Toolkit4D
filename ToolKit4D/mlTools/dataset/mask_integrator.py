# Peiyi Leng; edsml-pl1023
import numpy as np


def mask_integrator(indices_combination: tuple, agglomerate_masks: list):
    """
    Combine selected agglomerate masks into a single binary mask.

    This function takes a combination of indices and a list of agglomerate
    masks, and it combines the selected masks into a single binary mask by
    performing a logical OR operation across the selected masks.

    Args:
        indices_combination (tuple): A tuple of indices specifying which
                                     agglomerate masks to combine.
        agglomerate_masks (list): A list of binary masks, each representing an
                                  individual agglomerate.

    Returns:
        numpy.ndarray: A binary mask representing the combined result of the
                       selected agglomerate masks.
    """
    combined_mask = np.zeros_like(agglomerate_masks[0], dtype=bool)
    extracted_masks = [agglomerate_masks[index] for index in
                       indices_combination]
    for mask in extracted_masks:
        combined_mask = mask | combined_mask
    return combined_mask
