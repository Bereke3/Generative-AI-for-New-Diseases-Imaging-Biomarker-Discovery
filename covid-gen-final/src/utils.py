import torch
import numpy as np
import random
import os
from torchvision.utils import save_image

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_tensor_image(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensor = (tensor + 1) / 2
    save_image(tensor, path)

