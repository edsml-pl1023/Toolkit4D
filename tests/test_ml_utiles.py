from ToolKit4D.mlTools.utils import find_combination


def test_find_combination():
    assert len(find_combination(1)) == 1
    assert len(find_combination(5)) == 31
    assert len(find_combination(10)) == 1023
    assert len(find_combination(20)) == 1048575
