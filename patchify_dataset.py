import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# ------------------------------
# CONFIG
# ------------------------------
PATCH_SIZE = 512
STRIDE = 256   # 50% overlap
MIN_FOREGROUND_RATIO = 0.05  # keep tiles with >5% non-background pixels
BG_CLASS = 0

root = "data/FUSAR-Map"
sar_dir = os.path.join(root, "SAR_1024")
lbl_dir = os.path.join(root, "Labels_1024")

out_sar = os.path.join(root, f"SAR_{PATCH_SIZE}")
out_lbl = os.path.join(root, f"Labels_{PATCH_SIZE}")
os.makedirs(out_sar, exist_ok=True)
os.makedirs(out_lbl, exist_ok=True)

# ------------------------------
# FUNCTIONS
# ------------------------------

def patchify(img, size=PATCH_SIZE, stride=STRIDE):
    """Cut image into overlapping patches."""
    h, w = img.shape[:2]
    patches = []
    coords = []
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            patch = img[y:y+size, x:x+size]
            patches.append(patch)
            coords.append((y, x))
    return patches, coords


def is_useful_label(lbl):
    """Check if the patch has enough non-background pixels."""
    total = lbl.size
    non_bg = np.sum(lbl != BG_CLASS)
    ratio = non_bg / total
    return ratio >= MIN_FOREGROUND_RATIO


# ------------------------------
# MAIN LOOP
# ------------------------------

print("🔹 Starting smart patchification...")
num_saved = 0

# Loop through SAR images
for name in tqdm(sorted(os.listdir(sar_dir))):
    if not name.lower().endswith(".tif"):
        continue

    sar_path = os.path.join(sar_dir, name)

    # Match label file name
    lbl_name = name.replace("_SAR_", "_Label_")
    lbl_path = os.path.join(lbl_dir, lbl_name)

    if not os.path.exists(lbl_path):
        print(f"⚠️  Missing label for {name}")
        continue

    sar = np.array(Image.open(sar_path))
    lbl = np.array(Image.open(lbl_path))

    sar_patches, coords = patchify(sar)
    lbl_patches, _ = patchify(lbl)

    base = os.path.splitext(name)[0]
    base_lbl = base.replace("_SAR_", "_Label_")

    for i, (si, li) in enumerate(zip(sar_patches, lbl_patches)):
        # skip empty/background-only patches
        if not is_useful_label(li):
            continue

        sar_out = os.path.join(out_sar, f"{base}_p{i}.tif")
        lbl_out = os.path.join(out_lbl, f"{base_lbl}_p{i}.tif")

        Image.fromarray(si).save(sar_out)
        Image.fromarray(li).save(lbl_out)
        num_saved += 1

print(f"✅ Done! Saved {num_saved} useful patches in:")
print(f"   SAR: {out_sar}")
print(f"   Labels: {out_lbl}")
