import itertools


def find_combination(indices_length: int):
    """_summary_

    Args:
        indices_length (int): _description_
    """
    indices_combination = []
    for i in range(1, indices_length + 1):
        for j in itertools.combinations(range(indices_length), i):
            indices_combination.append(j)
    return indices_combination
