import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import FUSARMapDataset
import segmentation_models_pytorch as smp

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

root = "data/FUSAR-Map"
test_set = FUSARMapDataset(root, "test", augment=False)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,    
    in_channels=3,
    classes=5 
)

ckpt = torch.load("results/baseline/best_model.pth", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

# Colormap for 5 classes
colors = np.array([
    [0, 0, 0],        # Class 0: Background
    [255, 0, 0],      # Class 1: Buildings (red)
    [0, 255, 0],      # Class 2: Vegetation (green)
    [0, 0, 255],      # Class 3: Water (blue)
    [255, 255, 0]     # Class 4: Roads (yellow)
], dtype=np.uint8)

def decode_mask(mask):
    """Convert class indices to RGB image"""
    return colors[mask]

# Visualize 5 samples
for idx in [0, 5, 10, 15, 20]:
    img, mask = test_set[idx]
    img_t = img.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_t)
        pred = torch.argmax(torch.softmax(out, dim=1), dim=1).cpu().numpy()[0]

    # Convert from (C, H, W) to (H, W) for grayscale display
    # Since all 3 channels are identical (SAR duplicated), just take one
    img_np = img[0].cpu().numpy()  # Take first channel
    
    # Denormalize for better visualization (optional but recommended)
    # Reverse ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Denormalize using the first channel's stats (all channels are same for SAR)
    img_np = img_np * IMAGENET_STD[0] + IMAGENET_MEAN[0]
    img_np = np.clip(img_np, 0, 1)  # Clip to valid range

    # Create figure
    plt.figure(figsize=(15, 5))
    
    # SAR Input
    plt.subplot(1, 3, 1)
    plt.title("SAR Input")
    plt.imshow(img_np, cmap='gray')
    plt.axis('off')

    # Ground Truth
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(decode_mask(mask.numpy()))
    plt.axis('off')

    # Prediction
    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(decode_mask(pred))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

print("Visualization complete!")