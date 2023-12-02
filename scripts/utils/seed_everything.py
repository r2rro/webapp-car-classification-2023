import random
import os
import torch
import numpy as np

def seed_everything(seed: int=42):
    """
    Set seed for various random number generators to ensure reproducibility.

    Parameters:
        seed (int): Seed value for random number generators. Default is 42.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False