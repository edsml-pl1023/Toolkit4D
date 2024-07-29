import numpy as np


def mask_integrator(indices_combination: tuple, agglomerate_masks: list):
    combined_mask = np.zeros_like(agglomerate_masks[0], dtype=bool)
    extracted_masks = [agglomerate_masks[index] for index in
                       indices_combination]
    for mask in extracted_masks:
        combined_mask = mask | combined_mask
    return combined_mask
