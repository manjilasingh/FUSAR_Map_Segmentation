import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
from dataset_backu import FUSARMapDataset

# ---------------------------------------------
#  Metrics used in Shi et al. (2021)
# ---------------------------------------------
def compute_metrics(preds, targets, num_classes=5):
    preds = preds.flatten()
    targets = targets.flatten()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for i in range(len(preds)):
        cm[targets[i], preds[i]] += 1
    cm = cm.numpy()

    Ti = cm.sum(axis=1)
    pii = np.diag(cm)

    # Per-class Accuracy
    PA = np.divide(pii, Ti, out=np.zeros_like(pii, dtype=float), where=Ti != 0)
    # Overall Accuracy
    OA = pii.sum() / Ti.sum()
    # FWIoU
    Sij = cm
    FWIoU = np.sum(
        (Ti / Ti.sum()) * (pii / (Ti + Sij.sum(axis=0) - pii + 1e-6))
    )
    return OA, PA, FWIoU

# ---------------------------------------------
#  Dice + Weighted Cross Entropy Loss
# ---------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes).permute(0,3,1,2)
        intersection = torch.sum(probs * targets_onehot)
        union = torch.sum(probs + targets_onehot)
        dice = (2. * intersection + self.eps) / (union + self.eps)
        return 1 - dice

# ---------------------------------------------
#  Main training loop
# ---------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

root = "data/FUSAR-Map"
num_classes = 5

train_set = FUSARMapDataset(root, "train", augment=True)
val_set   = FUSARMapDataset(root, "val", augment=False)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=0)

# model
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=num_classes)
model = model.to(device)

# freeze encoder for first few epochs
for param in model.encoder.parameters():
    param.requires_grad = False

# optimizer + losses
ce_loss = nn.CrossEntropyLoss(weight=None)   # optional: pass weights here
dice_loss = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.93)

best_val_loss = float("inf")
patience = 15
wait = 0

for epoch in range(100):
    # ---- Unfreeze encoder after 5 epochs
    if epoch == 5:
        for param in model.encoder.parameters():
            param.requires_grad = True

    # ---- Training
    model.train()
    train_losses = []
    for imgs, masks in tqdm(train_loader, desc=f"Epoch [{epoch+1}/100] Train"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = ce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # ---- Validation
    model.eval()
    val_losses, preds_all, gts_all = [], [], []
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Epoch [{epoch+1}/100] Val"):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = ce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)
            val_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            preds_all.append(preds.cpu())
            gts_all.append(masks.cpu())

    train_loss = np.mean(train_losses)
    val_loss   = np.mean(val_losses)
    preds_all = torch.cat(preds_all)
    gts_all   = torch.cat(gts_all)

    OA, PA, FWIoU = compute_metrics(preds_all, gts_all, num_classes=num_classes)

    print(f"Epoch [{epoch+1}/100] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | OA: {OA:.4f} | FWIoU: {FWIoU:.4f}")
    print("Per-class Acc:", np.round(PA,3))

    # ---- Scheduler
    if (epoch+1) % 2 == 0:
        scheduler.step()

    # ---- Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"✅ Saved new best model (val loss: {val_loss:.4f})")
    else:
        wait += 1
        print(f"⚠️  No improvement ({wait}/{patience})")
        if wait >= patience:
            print("⛔ Early stopping triggered.")
            break

# ---------------------------------------------
#  Final results summary
# ---------------------------------------------
print("\n📊 Final Evaluation Results (compare to Shi et al., 2021)")
print("----------------------------------------------------------")
print(f"Overall Accuracy (OA): {OA*100:.2f}%")
print(f"Frequency Weighted IoU (FWIoU): {FWIoU*100:.2f}%")
for i, pa in enumerate(PA):
    print(f"Class {i} Accuracy (PA): {pa*100:.2f}%")
print("----------------------------------------------------------")
print("Paper baseline: OA ≈ 75.84%, FWIoU ≈ 66.59% (PT-Ours, DeepLabv3+-Xception)")
