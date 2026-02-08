import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import FUSARMapDataset
from tqdm import tqdm

# --- Setup device ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# --- Load datasets ---
root = "data/FUSAR-Map"
train_set = FUSARMapDataset(root, "train", augment=False)
val_set = FUSARMapDataset(root, "val", augment=False)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0)


# --- Model ---
model = smp.Unet(
    encoder_name="resnet152",
    encoder_weights="imagenet",
    in_channels=3,
    classes=5
).to(device)
    
# --- Loss and optimizer ---
# weights = torch.tensor([ 2.4780,  7.4606,  6.0717,  5.4192, 13.3554]).to(device)
ce_loss = torch.nn.CrossEntropyLoss()
dice_loss = smp.losses.DiceLoss(mode='multiclass')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
)


# --- Early stopping setup ---
best_val_loss = float("inf")
patience = 10   # stop if val loss doesn’t improve for 3 epochs
counter = 0
epochs = 100

def combined_loss(pred, target):
    return 0.7 * ce_loss(pred, target) + 0.3 * dice_loss(pred, target)

# --- Training loop ---
for epoch in range(epochs):
    # ---- TRAIN ----
    model.train()
    total_train_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] (Train)", leave=False)

    for img, mask in train_bar:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = combined_loss(output, mask.long())

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)

    # ---- VALIDATION ----
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] (Val)", leave=False)
        for img, mask in val_bar:
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            loss = combined_loss(output, mask.long())

            total_val_loss += loss.item()
            val_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_val_loss = total_val_loss / len(val_loader)


    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    scheduler.step(avg_val_loss)

    # ---- EARLY STOPPING ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), "best_unet_fusar.pth")
        print(f"✅ Saved new best model (val loss: {best_val_loss:.4f})")
    else:
        counter += 1
        print(f"⚠️  No improvement ({counter}/{patience})")

    if counter >= patience:
        print("⛔ Early stopping triggered.")
        break
