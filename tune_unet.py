import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import FUSARMapDataset
from tqdm import tqdm
import optuna

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

root = "data/FUSAR-Map"

# --- Define the objective function ---
def objective(trial):
    # --- Hyperparameters to tune ---
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    decoder_dropout = trial.suggest_float("decoder_dropout", 0.0, 0.5)
    encoder_name = trial.suggest_categorical("encoder_name", ["resnet18", "resnet34", "resnet50"])

    # --- Load datasets ---
    train_set = FUSARMapDataset(root, "train", augment=False)
    val_set = FUSARMapDataset(root, "val", augment=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Model ---
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=5,
        decoder_dropout=decoder_dropout
    ).to(device)

    # --- Loss and optimizer ---
    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Train for few epochs ---
    epochs = 10
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = ce_loss(output, mask.long())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                output = model(img)
                loss = ce_loss(output, mask.long())
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        trial.report(avg_val_loss, epoch)

        # early stop if too bad
        if trial.should_prune():
            raise optuna.TrialPruned()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    return best_val_loss


# --- Run the study ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15, timeout=None)

print("✅ Best trial:")
best_trial = study.best_trial
for key, value in best_trial.params.items():
    print(f"  {key}: {value}")
print(f"Best validation loss: {best_trial.value:.4f}")
