import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2, numpy as np, albumentations as A
from albumentations.pytorch import ToTensorV2


def add_speckle(img, intensity=0.2):
    noise = np.random.gamma(shape=1/intensity, scale=intensity, size=img.shape).astype(np.float32)
    out = img * noise
    return np.clip(out, 0, 255)


def get_transforms(augment=False):
    if augment:
        return A.Compose([
            # Safe geometric transforms only
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            
            # SAR-appropriate photometric augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.15, 
                contrast_limit=0.15, 
                p=0.3
            ),
            
            # Proper SAR speckle
            A.Lambda(image=lambda img, **kwargs: add_speckle(img, intensity=0.2), p=0.3),
            
            # Proper normalization (per image)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
        #     A.Lambda(
        #     image=lambda x, **kw: cv2.normalize(
        #         x, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32),
        #     p=1.0
        # ),
            # A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        #   A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),


            ToTensorV2()
        ])

class FUSARMapDataset(Dataset):
    def __init__(self, root, split, augment=False):
        self.root = root
        self.augment = augment
        with open(os.path.join(root, f"{split}.txt")) as f:
            self.names = [x.strip() for x in f.readlines()]

        self.sar_dir = os.path.join(root, "SAR_512")
        self.lbl_dir = os.path.join(root, "Labels_512")
        self.transforms = get_transforms(augment)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        base = self.names[idx]

        lbl_file = base.replace("_SAR_", "_Label_")

        sar = np.array(Image.open(os.path.join(self.sar_dir, base)))
        lbl = np.array(Image.open(os.path.join(self.lbl_dir, lbl_file)))

        # Convert single-channel SAR to 3-channel BEFORE augmentation
        if sar.ndim == 2:
            sar = np.stack([sar, sar, sar], axis=-1)  # [H, W] -> [H, W, 3]

    
        # If RGB mask, convert colors to class indices
        if lbl.ndim == 3:
            # Define class colors
            color_map = {
                (0, 0, 0): 0,
                (255, 0, 0): 1,
                (0, 255, 0): 2,
                (0, 0, 255): 3,
                (255, 255, 0): 4
            }

            # Create empty class map
            new_lbl = np.zeros((lbl.shape[0], lbl.shape[1]), dtype=np.int64)

            # Convert RGB pixels to class index
            for rgb, cls in color_map.items():
                mask = np.all(lbl == np.array(rgb), axis=-1)
                new_lbl[mask] = cls

            lbl = new_lbl  # replace with class map


        augmented = self.transforms(image=sar, mask=lbl)
        img = augmented["image"].float()
        mask = augmented["mask"].long()

        return img, mask
