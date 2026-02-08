import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

# ---------------------------------------------
#  NMT preprocessing (Shi et al., 2021)
# ---------------------------------------------
def nmt_preprocess(img, N=3.0):
    """Apply N-times-mean truncation (NMT) and scale to [0,255]."""
    img = img.astype(np.float32)
    nonzero = img[img > 0]
    if nonzero.size == 0:
        return np.zeros_like(img, dtype=np.uint8)
    mean_val = nonzero.mean()
    img = np.clip(img, 0, N * mean_val)
    img = (img / (N * mean_val)) * 255.0
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

# ---------------------------------------------
#  Optional speckle noise augmentation
# ---------------------------------------------
def add_speckle_noise(img, mean=0, var=0.04):
    noise = np.random.normal(mean, var**0.5, img.shape)
    noisy = img + img * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# ---------------------------------------------
#  Dataset class
# ---------------------------------------------
class FUSARMapDataset(Dataset):
    def __init__(self, root, split="train", augment=True, nmt=True):
        self.root = root
        self.split = split
        self.augment = augment
        self.nmt = nmt

        with open(os.path.join(root, f"{split}.txt")) as f:
            self.names = [x.strip() for x in f.readlines()]

        self.sar_dir = os.path.join(root, "SAR_512")
        self.mask_dir = os.path.join(root, "Labels_512")


        # Augmentation pipeline
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(var_limit=(10, 30), p=0.3),
                A.Blur(blur_limit=3, p=0.2),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        sar_path = os.path.join(self.sar_dir, f"{name}")
        m_dir= self.sar_dir.replace("_SAR_", "_Labels_")
        mask_path = os.path.join(m_dir, f"{name}")

        sar = cv2.imread(sar_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if sar is None:
            raise FileNotFoundError(
                f"Failed to load SAR image: {sar_path}\n"
                f"Check if file exists: {os.path.exists(sar_path)}"
            )

        if self.nmt:
            sar = nmt_preprocess(sar, N=3.0)

        # copy 1→3 channels
        if sar.ndim == 2:
            sar = np.stack([sar] * 3, axis=-1)

        # optional speckle noise
        if np.random.rand() < 0.2:
            sar = add_speckle_noise(sar)

        # apply augmentations
        if self.transform:
            aug = self.transform(image=sar, mask=mask)
            sar, mask = aug["image"], aug["mask"]

        # Normalize to [0,1] tensor
        sar = sar.astype(np.float32) / 255.0
        sar = np.transpose(sar, (2, 0, 1))  # (C,H,W)
        mask = mask.astype(np.int64)

        return torch.tensor(sar), torch.tensor(mask)
