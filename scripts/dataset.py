import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class KlaDataset(Dataset):
    def __init__(self, gt_dir, lr_dir, patch_size=128, augment=True):
        self.gt_dir = gt_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.augment = augment
        
        self.files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.npy')])
        print(f"Found {len(self.files)} image pairs")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        
        gt = np.load(os.path.join(self.gt_dir, filename)).astype(np.float32)
        lr = np.load(os.path.join(self.lr_dir, filename)).astype(np.float32)
        
        gt = gt / 255.0
        lr = lr / 255.0
        
        if self.augment:
            if random.random() > 0.5:
                gt = np.flip(gt, axis=0).copy()
                lr = np.flip(lr, axis=0).copy()
            if random.random() > 0.5:
                gt = np.flip(gt, axis=1).copy()
                lr = np.flip(lr, axis=1).copy()
            if random.random() > 0.5:
                gt = np.rot90(gt).copy()
                lr = np.rot90(lr).copy()
        
        gt = torch.from_numpy(gt).unsqueeze(0).float()
        lr = torch.from_numpy(lr).unsqueeze(0).float()
        
        return lr, gt