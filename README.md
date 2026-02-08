# FUSAR-Map: Semantic Segmentation of SAR Images using U-Net

This project implements a deep learning-based semantic segmentation system for Synthetic Aperture Radar (SAR) imagery using U-Net architecture with ResNet encoders. The model performs pixel-level classification of SAR images into 5 land cover classes: Water, Road, Building, Vegetation, and Others.

## Acknowledgments

Special thanks to **Bhavya Duvvuri** for the initial idea and guidance during the development of this project.

## Features

- **U-Net Architecture**: Uses segmentation_models_pytorch with ResNet encoders (ResNet34/ResNet152)
- **SAR-Specific Augmentations**: Includes speckle noise simulation and other SAR-appropriate data augmentations
- **Advanced Loss Functions**: Combines Cross-Entropy, Dice Loss, and Focal Loss for better class imbalance handling
- **Comprehensive Evaluation**: Computes Overall Accuracy (OA), Per-Class Accuracy (PA), and Frequency Weighted IoU (FWIoU)
- **Data Preprocessing**: Smart patchification with overlap handling and foreground filtering

## Dataset Structure

The project expects the FUSAR-Map dataset in the following structure:

```
data/FUSAR-Map/
├── SAR_512/          # SAR image patches (512x512)
├── Labels_512/       # Label patches (512x512)
├── SAR_1024/         # Original SAR images (1024x1024)
├── Labels_1024/      # Original labels (1024x1024)
├── train.txt         # Training set file list
├── val.txt           # Validation set file list
└── test.txt          # Test set file list
```

## Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (optional, but recommended) or Apple Silicon (MPS support)

### Install Dependencies

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations
pip install pillow numpy matplotlib seaborn tqdm
```

Or create a requirements file and install:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

#### Create Patches from Original Images

If you have 1024x1024 images and need to create 512x512 patches:

```bash
python patchify_dataset.py
```

This script:
- Splits 1024x1024 images into 512x512 patches with 50% overlap
- Filters out patches with less than 5% foreground pixels
- Saves patches to `SAR_512/` and `Labels_512/` directories

#### Create Train/Val/Test Splits

Generate train/validation/test splits (70/15/15):

```bash
python create_split.py
```

This creates `train.txt`, `val.txt`, and `test.txt` files listing the image filenames for each split.

### 2. Training

Train the model with data augmentation:

```bash
python train.py
```

**Training Configuration:**
- **Model**: U-Net with ResNet34 encoder, ImageNet pretrained weights
- **Input**: 1-channel SAR images (converted to 3-channel for compatibility)
- **Output**: 5-class segmentation masks
- **Loss**: Combined loss (50% CE + 30% Dice + 20% Focal)
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau
- **Batch Size**: 4
- **Epochs**: 30
- **Warmup**: First 2 epochs with frozen encoder

The best model will be saved to `results/augmented/best_model.pth` based on validation loss.

### 3. Evaluation

Evaluate the trained model on the test set:

```bash
python evaluate_metrics.py
```

This script:
- Loads the trained model
- Computes confusion matrix
- Calculates Overall Accuracy (OA), Per-Class Accuracy (PA), and FWIoU
- Generates visualization plots:
  - `confusion_matrix.png`
  - `per_class_accuracy.png`
  - `comparison_baseline.png`

**Note**: Update the model path in `evaluate_metrics.py` to point to your trained model checkpoint.

### 4. Visualization

Visualize predictions on test samples:

```bash
python visualize.py
```

This displays side-by-side comparisons of:
- SAR input image
- Ground truth segmentation
- Model predictions

**Note**: Update the model path and checkpoint loading code in `visualize.py` to match your saved model format.

### 5. Testing

Run inference on test images:

```bash
python test.py
```

## Project Structure

```
FUSAR-Map/
├── data/
│   └── FUSAR-Map/          # Dataset directory
├── results/                 # Saved model checkpoints
│   ├── augmented/
│   ├── baseline/
│   └── debug/
├── dataset.py              # Dataset class with augmentations
├── train.py                # Main training script
├── test.py                 # Testing script
├── evaluate_metrics.py     # Evaluation and metrics computation
├── visualize.py            # Visualization utilities
├── patchify_dataset.py     # Create patches from large images
├── create_split.py         # Generate train/val/test splits
└── README.md               # This file
```

## Model Architecture

- **Encoder**: ResNet34 or ResNet152 (pretrained on ImageNet)
- **Decoder**: U-Net decoder with skip connections
- **Input**: 3-channel images (SAR single-channel duplicated)
- **Output**: 5-class logits (Water, Road, Building, Vegetation, Others)

## Data Augmentation

The training pipeline includes SAR-specific augmentations:

- **Geometric**: Horizontal/Vertical flips, 90° rotations
- **Photometric**: Random brightness/contrast adjustments
- **SAR-Specific**: Speckle noise simulation using Gamma distribution

## Loss Function

The model uses a combined loss function:

```
Loss = 0.5 × CrossEntropy + 0.3 × DiceLoss + 0.2 × FocalLoss
```

This combination helps handle class imbalance common in remote sensing datasets.

## Evaluation Metrics

- **Overall Accuracy (OA)**: Percentage of correctly classified pixels
- **Per-Class Accuracy (PA)**: Accuracy for each individual class
- **Frequency Weighted IoU (FWIoU)**: IoU weighted by class frequency

## Device Support

The code automatically detects and uses:
- **MPS** (Metal Performance Shaders) on Apple Silicon Macs
- **CUDA** on NVIDIA GPUs (if available)
- **CPU** as fallback

## Notes

- The dataset uses RGB color-coded labels that are converted to class indices (0-4)
- Single-channel SAR images are converted to 3-channel by duplication for compatibility with ImageNet-pretrained encoders
- Model checkpoints are saved based on best validation loss
- The encoder is frozen for the first 2 epochs as a warmup strategy

## License

[Specify your license here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add your citation here]
```

