from ToolKit4D.mlTools.utils import find_combination
from ToolKit4D.mlTools.utils import get_top_combination


def test_find_combination():
    assert len(find_combination(1)) == 1
    assert len(find_combination(5)) == 31
    assert len(find_combination(10)) == 1023
    assert len(find_combination(20)) == 1048575


def test_get_top_combination():
    # Sample agglomerate_comb dictionary
    agglomerate_comb = {
        (0,): 0.95,
        (1,): 0.90,
        (2,): 0.86,
        (3,): 0.88,
        (4,): 0.74,
        (0, 1): 0.90,
        (1, 2): 0.91,
        (2, 3): 0.87,
        (3, 4): 0.83,
        (0, 2): 0.89,
        (1, 3): 0.90,
        (2, 4): 0.79,
        (0, 3): 0.64,
        (1, 4): 0.55,
        (0, 4): 0.91,
        (1, 2, 3): 0.75,
        (2, 3, 4): 0.74,
        (0, 1, 2, 3): 0.60,  # Lower probability for 4-element combination
        (0, 3, 4): 0.73,
        (0, 1, 3): 0.72,
        (0, 1, 4): 0.71
    }

    # Number of top agglomerates to find
    num_agglomerates = 5

    # Theoretical expected result
    expected_result = [
        (0,),
        (1, 2),
        (3,),
        (4,),
    ]

    # Call the function with the sample data
    top_combinations = get_top_combination(agglomerate_comb, num_agglomerates)

    # Assert the results
    assert top_combinations == expected_result
