import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import FUSARMapDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# ---- Metrics ----
def compute_confusion_matrix(preds, labels, num_classes):
    mask = (labels >= 0) & (labels < num_classes)
    hist = np.bincount(
        num_classes * labels[mask].astype(int) + preds[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist

def compute_metrics(conf_matrix):
    # Overall Accuracy (OA)
    OA = np.diag(conf_matrix).sum() / conf_matrix.sum()

    # Per-class Accuracy (PA)
    PA = np.diag(conf_matrix) / (conf_matrix.sum(axis=1) + 1e-10)

    # Frequency Weighted IoU (FWIoU)
    freq = conf_matrix.sum(axis=1) / conf_matrix.sum()
    iu = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix) + 1e-10)
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()

    return OA, PA, FWIoU

# ---- Device ----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# ---- Dataset ----
root = "data/FUSAR-Map"
num_classes = 5
test_set = FUSARMapDataset(root, "test", augment=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# ---- Model ----
model = smp.Unet(
    encoder_name="resnet152",
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes
).to(device)

model.load_state_dict(torch.load("best_unet_fusar_with_resnet152.pth", map_location=device))
model.eval()

# ---- Evaluation ----
conf_matrix = np.zeros((num_classes, num_classes))

with torch.no_grad():
    for imgs, masks in tqdm(test_loader, desc="Evaluating"):
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        labels = masks.cpu().numpy()
        conf_matrix += compute_confusion_matrix(preds.flatten(), labels.flatten(), num_classes)

OA, PA, FWIoU = compute_metrics(conf_matrix)

# ---- Print results ----
print("\n📊 Evaluation Results for best_model.pth")
print("----------------------------------------------------------")
print(f"Overall Accuracy (OA): {OA * 100:.2f}%")
print(f"Frequency Weighted IoU (FWIoU): {FWIoU * 100:.2f}%")
for i, pa in enumerate(PA):
    print(f"Class {i} Accuracy (PA): {pa * 100:.2f}%")
print("----------------------------------------------------------")
print("Paper baseline (Shi et al., 2021): OA ≈ 75.84%, FWIoU ≈ 66.59%")


# ---- Plot Confusion Matrix ----
classes = ["Others", "Building", "Vegetation", "Water", "Road"]

plt.figure(figsize=(7,6))
sns.heatmap(conf_matrix / conf_matrix.sum(axis=1, keepdims=True), 
            annot=True, fmt=".2f", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title(f"Normalized Confusion Matrix\n(OA={OA*100:.2f}%, FWIoU={FWIoU*100:.2f}%)")
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# ---- Plot Per-Class Accuracy (PA) ----
plt.figure(figsize=(7,4))
plt.bar(classes, PA*100, color="royalblue", edgecolor="black")
plt.title("Per-Class Accuracy (PA)")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
for i, v in enumerate(PA*100):
    plt.text(i, v+1, f"{v:.1f}%", ha="center")
plt.tight_layout()
plt.savefig("per_class_accuracy.png", dpi=300)
plt.show()

# Paper baseline from Table IV in Shi et al., 2021
paper_PA = np.array([86.42, 10.42, 0.06, 0.09, 0.37])  # Replace with your earlier baseline if desired

plt.figure(figsize=(7,4))
x = np.arange(len(classes))
width = 0.35
plt.bar(x - width/2, PA*100, width, label='Your Model', color='royalblue')
plt.bar(x + width/2, paper_PA, width, label='Shi et al. (2021)', color='lightgray')
plt.xticks(x, classes)
plt.ylabel("Accuracy (%)")
plt.title("Per-Class Accuracy Comparison")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig("comparison_baseline.png", dpi=300)
plt.show()
