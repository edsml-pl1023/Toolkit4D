# Peiyi Leng; edsml-pl1023
import torch
import random
import numpy as np


def set_seed(seed):
    """
    Set the seed for all relevant random number generators to ensure
    reproducibility.

    This function sets the seed for Python's `random` module, NumPy, and
    PyTorch to ensure consistent and reproducible results across different
    runs. It also configures PyTorch's CUDA settings to avoid non-deterministic
    behavior.

    Args:
        seed (int): The seed value to use for all random number generators.

    Returns:
        bool: Returns `True` after the seed has been set successfully.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True
