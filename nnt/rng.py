import random

import numpy as np
import torch


def seed_all(seed=1337):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
