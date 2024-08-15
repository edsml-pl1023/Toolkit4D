from ...stages.separate_rocks import binary_search_agglomerates
from .predict_NumAgglomerates import predict_NumAgglomerates


def recursive_agglomerate_search(guess, min_obj_size, initial_mask,
                                 max_agglomerates=14, max_depth=2,
                                 current_depth=0):
    # Perform binary search agglomeration
    agglomerate_masks = binary_search_agglomerates(
        guess, min_obj_size, initial_mask
    )

    # List to store results
    final_agglomerates = []

    # Iterate through each agglomerate mask in the list
    for agglomerate in agglomerate_masks:
        # Predict the number of agglomerates in this mask
        num_agglomerates = predict_NumAgglomerates(agglomerate)

        # If the number of agglomerates is 1, we stop the recursion
        # for this mask
        if num_agglomerates == 1:
            final_agglomerates.append(agglomerate)
        else:
            # Otherwise, if we haven't reached the max depth,
            # recursively apply the function
            if current_depth < max_depth:
                next_agglomerates = recursive_agglomerate_search(
                    num_agglomerates, min_obj_size, agglomerate,
                    max_agglomerates, max_depth, current_depth + 1
                )
                final_agglomerates.extend(next_agglomerates)
            else:
                # If max depth is reached, add the current agglomerate
                # without further splitting
                final_agglomerates.append(agglomerate)

        # If we have reached or exceeded the maximum number
        # of agglomerates, stop and return
        if len(final_agglomerates) >= max_agglomerates:
            return final_agglomerates[:max_agglomerates]

    return final_agglomerates
