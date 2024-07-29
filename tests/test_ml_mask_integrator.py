from ToolKit4D.pipeline import ToolKitPipeline
from ToolKit4D.mlTools.utils import find_combination
import numpy as np


def test_mask_overlap():
    LA2d0 = ToolKitPipeline(
        './tests/test_data/LA2_d0_v1_uint16_unnormalised_254_254_254.raw'
        )
    LA2d0.separate_rocks()
    for indices_combination in find_combination(len(LA2d0.agglomerate_masks)):
        if len(indices_combination) != 1:
            extracted_masks = [LA2d0.agglomerate_masks[index]
                               for index in indices_combination]
            combined_mask = np.ones_like(extracted_masks[0], dtype=bool)
            for mask in extracted_masks:
                combined_mask = mask & combined_mask
            assert np.sum(combined_mask) == 0
