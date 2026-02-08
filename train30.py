import os
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

train_set = FUSARMapDataset(root, "train", augment=False)
val_set   = FUSARMapDataset(root, "val", augment=False)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=16, num_workers=0)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=5
).to(device)

encoder_params = list(model.encoder.parameters())
decoder_params = [p for n,p in model.named_parameters() if not n.startswith("encoder.")]

optimizer = torch.optim.Adam([
    {"params": encoder_params, "lr": 1e-4},
    {"params": decoder_params, "lr": 3e-4}
], weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=5
# )
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)


checkpoint_path = "results/baseline/best_model.pth"

# ============ RESUME SUPPORT ============
start_epoch = 0
best_val_loss = float("inf")

if os.path.exists(checkpoint_path):
    print("🔄 Found checkpoint — loading...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        # New format
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
        scheduler.load_state_dict(checkpoint["sched_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_loss"]
        print(f"➡️ Resumed at epoch {start_epoch}, best loss {best_val_loss:.4f}")

    else:
        # Old format (only model weights)
        print("⚠️ Checkpoint is old format (model only). Loading model weights only.")
        model.load_state_dict(checkpoint)
        start_epoch = 0
        best_val_loss = float("inf")

else:
    print("🆕 Training from scratch")

# def compute_class_weights(dataloader, num_classes=5):
#     """
#     Computes class weights based on inverse pixel frequency.
#     """
#     print("📊 Calculating class pixel frequencies...")
#     class_counts = np.zeros(num_classes, dtype=np.int64)

#     # Loop through a subset or whole dataset
#     for imgs, masks in tqdm(dataloader):
#         masks = masks.numpy()
#         for c in range(num_classes):
#             class_counts[c] += np.sum(masks == c)

#     print("Class pixel counts:", class_counts)

#     # Avoid division by zero
#     class_counts = np.maximum(class_counts, 1)

#     # Inverse frequency
#     inv_freq = 1.0 / class_counts

#     # Normalize so weights sum to num_classes
#     weights = inv_freq / np.sum(inv_freq) * num_classes

#     print("✅ Computed class weights:", weights)
#     return torch.tensor(weights, dtype=torch.float32, device=device)


# ============ LOSSES ============
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.softmax(pred, dim=1)
    target_onehot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1])
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()

    intersection = (pred * target_onehot).sum(dim=(0,2,3))
    union = pred.sum(dim=(0,2,3)) + target_onehot.sum(dim=(0,2,3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def focal_loss(inputs, targets, alpha=1, gamma=2):
    ce = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce)
    loss = alpha * ((1 - pt) ** gamma) * ce
    return loss.mean()

# weights = compute_class_weights(train_loader, num_classes=5)

weights = torch.tensor([0.2668174, 1.03134972, 0.80046859, 0.6969733, 2.20439098], dtype=torch.float32, device=device)
criterion_ce = nn.CrossEntropyLoss(weight=weights)
def combined_loss(pred, target):
    ce = criterion_ce(pred, target)
    dice = dice_loss(pred, target)
    focal = focal_loss(pred, target)
    return 0.5 * ce + 0.3 * dice + 0.2 * focal

def compute_iou(pred, target, num_classes=5):
    pred = torch.argmax(torch.softmax(pred, dim=1), dim=1).cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    ious = []

    for cls in range(num_classes):
        pred_mask = pred == cls
        true_mask = target == cls
        inter = np.logical_and(pred_mask, true_mask).sum()
        uni = np.logical_or(pred_mask, true_mask).sum()
        ious.append(np.nan if uni == 0 else inter / uni)
    return np.array(ious), np.nanmean(ious)

# ============ TRAIN SETTINGS ============
freeze_epochs = 2
total_epochs = 120

if start_epoch == 0:
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("❄️  Encoder frozen for warm-up")

# ============ TRAIN LOOP ============
for epoch in range(start_epoch, total_epochs):
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

    # ===== validation =====
    model.eval()
    val_loss = 0
    ious_list = []
    print_once = True  
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            out = model(img)
            # ✅ print only once per epoch
            if print_once:
                pred = torch.argmax(out, dim=1)
                unique, count = np.unique(pred.cpu(), return_counts=True)
                print("Predicted class distribution:", dict(zip(unique, count)))
                print_once = False  # don't print again this epoch
            
            val_loss += combined_loss(out, mask).item()
            _, miou = compute_iou(out, mask)
            ious_list.append(miou)

    val_loss /= len(val_loader)
    mean_iou = np.nanmean(ious_list)

    scheduler.step()
    print(f"✅ Epoch {epoch} | Val Loss={val_loss:.4f} | mIoU={mean_iou:.4f}")

    # unfreeze encoder
    if epoch == freeze_epochs:
        for param in model.encoder.parameters():
            param.requires_grad = True
        print("🔓 Encoder unfrozen — full fine-tuning")

    # save improved model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "best_loss": best_val_loss
        }, checkpoint_path)
        print(f"💾 Saved checkpoint (best loss {best_val_loss:.4f})")
