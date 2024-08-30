# Peiyi Leng; edsml-pl1023
from ...stages.separate_rocks import binary_search_agglomerates
from .predict_NumAgglomerates import predict_NumAgglomerates
import numpy as np


def recursive_agglomerate_search(guess, min_obj_size, initial_mask,
                                 model, max_agglomerates=14,
                                 max_depth=2, current_depth=0):
    """
    Recursively search and separate rock agglomerates in a 3D binary image 
    using a model to predict the number of agglomerates.

    This function performs a recursive search to separate agglomerates in a 
    3D binary image. It uses a trained model to predict the number of 
    agglomerates in each mask and recursively refines the masks until a single 
    agglomerate is found or the maximum recursion depth is reached.

    Args:
        guess (int): Initial guess for the number of agglomerates to search for.
        min_obj_size (int): The minimum size (in pixels) for an agglomerate to 
                            be considered valid.
        initial_mask (numpy.ndarray): The initial binary mask of the 3D image 
                                      where agglomerates are marked.
        model (torch.nn.Module): The trained PyTorch model used to predict the 
                                 number of agglomerates.
        max_agglomerates (int, optional): The maximum number of agglomerates to 
                                          return. Defaults to 14.
        max_depth (int, optional): The maximum recursion depth. Defaults to 2.
        current_depth (int, optional): The current recursion depth. Defaults to 0.

    Returns:
        list[numpy.ndarray]: A list of binary masks, each representing one of 
                             the largest detected agglomerates, up to the 
                             specified maximum.
    """
    # Perform binary search agglomeration
    agglomerate_masks = binary_search_agglomerates(
        guess, min_obj_size, initial_mask
    )

    # List to store results
    final_agglomerates = []

    # Iterate through each agglomerate mask in the list
    for agglomerate in agglomerate_masks:
        print(f'\t -- At depth {current_depth}')
        # Predict the number of agglomerates in this mask
        num_agglomerates = predict_NumAgglomerates(agglomerate)
        print(f'\t -- ML detect {num_agglomerates} agglomerates')

        # If the number of agglomerates is 1, we stop the recursion
        # for this mask
        if num_agglomerates == 1:
            final_agglomerates.append(agglomerate)
        else:
            # Otherwise, if we haven't reached the max depth,
            # recursively apply the function
            if current_depth < max_depth:
                next_agglomerates = recursive_agglomerate_search(
                    num_agglomerates, min_obj_size, agglomerate, model,
                    max_agglomerates, max_depth, current_depth + 1
                )
                final_agglomerates.extend(next_agglomerates)
            else:
                # If max depth is reached, add the current agglomerate
                # without further splitting
                final_agglomerates.append(agglomerate)

    # If this is the top-level call (depth 0), sort and return the
    # largest agglomrates
    if current_depth == 0:
        final_agglomerates_sorted = sorted(
            final_agglomerates, key=lambda x: np.sum(x), reverse=True)
        return final_agglomerates_sorted[:max_agglomerates]
    else:
        return final_agglomerates
