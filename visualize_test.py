import torch
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import FUSARMapDataset

# --- Colormap for 5 classes ---
colors = np.array([
    [0, 0, 0],        # Class 0: black (background)
    [255, 0, 0],      # Class 1: red building
    [0, 255, 0],      # Class 2: green vegetation
    [0, 0, 255],      # Class 3: blue water
    [255, 255, 0]     # Class 4: yellow road
], dtype=np.uint8)

def decode_mask(mask):
    """Convert class indices (H, W) to RGB mask."""
    return colors[mask]

# --- Device setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# --- Load test dataset ---
root = "data/FUSAR-Map"
test_set = FUSARMapDataset(root, "test", augment=False)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=0)

# --- Load model ---
model = smp.Unet(
    encoder_name="resnet152",
    encoder_weights=None,   # no need for imagenet when loading trained weights
    in_channels=3,
    classes=5
).to(device)

# --- Load trained weights ---
model.load_state_dict(torch.load("best_unet_fusar_with_resnet152.pth", map_location=device))
model.eval()

start_idx, end_idx = 43, 48
# num_samples = min(5, len(test_set))

# --- Visualize first 5 test images ---
num_samples = min(5, len(test_set))
with torch.no_grad():
    for i in range(start_idx, min(end_idx, len(test_set))):
    # for i in range(num_samples):

        img, mask = test_set[i]
        print(img.min().item(), img.max().item())

        img, mask = img.unsqueeze(0).to(device), mask.to(device)

        # Inference
        output = model(img)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Convert to numpy
        img_np = img[0].permute(1, 2, 0).cpu().numpy()
        mask_np = mask.cpu().numpy()

        # Plot
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow((img_np - img_np.min()) / (img_np.max() - img_np.min()))
        # plt.imshow(img_np)
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(decode_mask(mask_np))
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(decode_mask(pred))
        plt.title("Prediction")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
