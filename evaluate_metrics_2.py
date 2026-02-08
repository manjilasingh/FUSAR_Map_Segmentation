import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import FUSARMapDataset
from tqdm import tqdm

def analyze_class_distribution(masks, num_classes=5):
    """Analyze pixel distribution per class"""
    masks_flat = masks.flatten()
    total_pixels = len(masks_flat)
    
    print("\n" + "="*60)
    print("📊 CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    for c in range(num_classes):
        count = (masks_flat == c).sum()
        percentage = (count / total_pixels) * 100
        print(f"Class {c}: {count:,} pixels ({percentage:.2f}%)")
    print("="*60)

def compute_detailed_metrics(preds, targets, num_classes=5):
    """Compute detailed per-class metrics"""
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    
    # Confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for c_true in range(num_classes):
        for c_pred in range(num_classes):
            conf_matrix[c_true, c_pred] = ((targets_flat == c_true) & (preds_flat == c_pred)).sum()
    
    print("\n" + "="*60)
    print("📈 DETAILED PER-CLASS METRICS")
    print("="*60)
    
    for c in range(num_classes):
        TP = conf_matrix[c, c]
        FP = conf_matrix[:, c].sum() - TP  # Predicted as c but not c
        FN = conf_matrix[c, :].sum() - TP  # Actually c but predicted as other
        TN = conf_matrix.sum() - TP - FP - FN
        
        # Metrics
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)  # Same as PA
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        iou = TP / (TP + FP + FN + 1e-10)
        
        total_class_pixels = conf_matrix[c, :].sum()
        
        print(f"\nClass {c} ({total_class_pixels:,} pixels):")
        print(f"  ✓ True Positives:  {TP:,}")
        print(f"  ✗ False Positives: {FP:,}")
        print(f"  ✗ False Negatives: {FN:,}")
        print(f"  → Precision: {precision*100:.2f}% (of predictions, how many correct?)")
        print(f"  → Recall:    {recall*100:.2f}% (of ground truth, how many found?)")
        print(f"  → F1-Score:  {f1*100:.2f}%")
        print(f"  → IoU:       {iou*100:.2f}%")
    
    print("="*60)
    
    return conf_matrix

def visualize_errors(img, pred, target, num_classes=5):
    """Visualize where the model makes mistakes"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Colormap
    colors = np.array([
        [0, 0, 0],        # Class 0 - Black
        [255, 0, 0],      # Class 1 - Red
        [0, 255, 0],      # Class 2 - Green
        [0, 0, 255],      # Class 3 - Blue
        [255, 255, 0]     # Class 4 - Yellow
    ], dtype=np.uint8)
    
    def apply_colormap(mask):
        return colors[mask]
    
    # Original image
    if img.shape[0] == 3:  # (C, H, W) -> (H, W, C)
        img = np.transpose(img, (1, 2, 0))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis('off')
    
    # Ground truth
    axes[0, 1].imshow(apply_colormap(target))
    axes[0, 1].set_title("Ground Truth")
    axes[0, 1].axis('off')
    
    # Prediction
    axes[0, 2].imshow(apply_colormap(pred))
    axes[0, 2].set_title("Prediction")
    axes[0, 2].axis('off')
    
    # Error map (where pred != target)
    error_map = (pred != target).astype(np.uint8)
    axes[1, 0].imshow(error_map, cmap='hot')
    axes[1, 0].set_title(f"Error Map ({error_map.sum():,} wrong pixels)")
    axes[1, 0].axis('off')
    
    # Per-class errors
    for c in range(min(2, num_classes-1)):  # Show errors for classes 1 and 2
        class_mask = (target == c+1)
        class_errors = class_mask & (pred != target)
        axes[1, c+1].imshow(class_errors, cmap='Reds')
        axes[1, c+1].set_title(f"Class {c+1} Errors ({class_errors.sum():,} pixels)")
        axes[1, c+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved error visualization to 'error_analysis.png'")
    plt.close()

# ==================== MAIN DIAGNOSTIC SCRIPT ====================

if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Configuration
    root = "data/FUSAR-Map"
    num_classes = 5
    model_path = "best_unet_fusar_with_resnet152.pth"
    encoder_name = "resnet152"

    # Load dataset
    test_set = FUSARMapDataset(root, "test", augment=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    
    # Load model
    model = smp.Unet(
        encoder_name='resnet152',
        encoder_weights=None,
        in_channels=3,
        classes=num_classes
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Collect all predictions
    print("Running inference on test set...")
    all_preds = []
    all_targets = []
    sample_img = None
    sample_pred = None
    sample_target = None
    
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(tqdm(test_loader)):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks = masks.cpu().numpy()
            
            all_preds.append(preds)
            all_targets.append(masks)
            
            # Save first sample for visualization
            if i == 0:
                sample_img = imgs[0].cpu().numpy()
                sample_pred = preds[0]
                sample_target = masks[0]
    
    # Concatenate all results
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Analysis
    analyze_class_distribution(all_targets, num_classes)
    conf_matrix = compute_detailed_metrics(all_preds, all_targets, num_classes)
    
    # Visualize errors on first sample
    visualize_errors(sample_img, sample_pred, sample_target, num_classes)
    
    print("\n" + "="*60)
    print("💡 INSIGHTS:")
    print("="*60)
    print("If you see:")
    print("  • High FP: Model over-predicts this class (false alarms)")
    print("  • High FN: Model under-predicts this class (misses)")
    print("  • Low IoU but high Recall: Boundary issues / over-segmentation")
    print("  • Huge class imbalance: Consider weighted loss or focal loss")
    print("="*60)