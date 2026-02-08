import torch, torch.nn as nn
from torch.utils.data import DataLoader
from dataset import FUSARMapDataset
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

root = "data/FUSAR-Map"

train_set = FUSARMapDataset(root, "train", augment=True)  # baseline first
val_set   = FUSARMapDataset(root, "val", augment=False)

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=4)

model = smp.Unet(
    encoder_name="resnet34",     # encoder backbone
    encoder_weights="imagenet",  # pretrained on ImageNet
    in_channels=1,               # SAR has 1 channel
    classes=5                    # your number of classes
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)



best_val_loss = float("inf")
save_path = "results/augmented/best_model.pth"


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.softmax(pred, dim=1)  # [B, C, H, W]
    target_onehot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1])
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

    intersection = (pred * target_onehot).sum(dim=(0,2,3))
    union = pred.sum(dim=(0,2,3)) + target_onehot.sum(dim=(0,2,3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def focal_loss(inputs, targets, alpha=1, gamma=2):
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    loss = alpha * ((1 - pt) ** gamma) * ce_loss
    return loss.mean()

criterion_ce = nn.CrossEntropyLoss()

def combined_loss(pred, target):
    ce = criterion_ce(pred, target)
    dice = dice_loss(pred, target)
    focal = focal_loss(pred, target)
    return 0.5 * ce + 0.3 * dice + 0.2 * focal


def compute_iou(pred, target, num_classes=5):
    pred = torch.argmax(torch.softmax(pred, dim=1), dim=1)
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()

    ious = []
    for cls in range(num_classes):
        pred_mask = pred == cls
        true_mask = target == cls

        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()

        if union == 0:
            ious.append(np.nan)  # ignore class not present
        else:
            ious.append(intersection / union)

    return np.array(ious), np.nanmean(ious)
# Freeze encoder first 2–3 epochs (warmup)
for param in model.encoder.parameters():
    param.requires_grad = False
freeze_epochs = 2

for epoch in range(30):  # start with 5 to test, then 20
    model.train()
    pbar = tqdm(train_loader)
    for img, mask in pbar:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = combined_loss(out, mask)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch {epoch} Loss {loss.item():.4f}")

    # simple val loop
    model.eval()
    val_loss = 0
    ious_list = []
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            out = model(img)

            val_loss += combined_loss(out, mask).item()

            # compute IoU
            ious, mean_iou = compute_iou(out, mask)
            ious_list.append(mean_iou)

    val_loss /= len(val_loader)
    miou = np.nanmean(np.array(ious_list))
    scheduler.step(val_loss)

    print(f"Val Loss: {val_loss:.4f}, mIoU: {miou:.4f}")
    if epoch == freeze_epochs:
        for param in model.encoder.parameters():
            param.requires_grad = True
        print("🔓 Encoder unfrozen — fine-tuning full network")

    # After validation
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print(f"✅ Saved new best model at epoch {epoch}")
