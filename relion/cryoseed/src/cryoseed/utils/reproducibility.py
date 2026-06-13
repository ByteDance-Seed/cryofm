import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): base seed
        deterministic (bool): enforce deterministic cudnn behavior
    """

    # Python
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Hash seed (for dataloader workers etc.)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False