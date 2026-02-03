import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class ZBinSlicesDataset(Dataset):
    def __init__(self, data_dir, manifest_path, image_size=256):
        self.data_dir = data_dir
        self.image_size = image_size

        with open(manifest_path, "r") as f:
            self.items = [line.strip().split(",") for line in f]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, zbin = self.items[idx]
        zbin = int(zbin)

        path = os.path.join(self.data_dir, fname)
        img = np.load(path)

        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32)

        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = img * 2.0 - 1.0

        img = torch.from_numpy(img).unsqueeze(0)

        return {
            "pixel_values": img,
            "class_labels": torch.tensor(zbin, dtype=torch.long)
        }

