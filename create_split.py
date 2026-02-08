import os, random

# Paths
img_dir = "data/FUSAR-Map/SAR_512"
save_dir = "data/FUSAR-Map/"

# List all images
images = sorted([f for f in os.listdir(img_dir) if f.endswith(".tif")])

# Shuffle for randomness
random.seed(42)
random.shuffle(images)

# Compute split sizes
n = len(images)
train_size = int(0.7 * n)
val_size = int(0.15 * n)
test_size = n - train_size - val_size


train_files = images[:train_size]
val_files = images[train_size:train_size+val_size]
test_files = images[train_size+val_size:train_size+val_size+test_size]

# Write lists to txt files
with open(os.path.join(save_dir, "train.txt"), "w") as f:
    f.writelines([f"{img}\n" for img in train_files])

with open(os.path.join(save_dir, "val.txt"), "w") as f:
    f.writelines([f"{img}\n" for img in val_files])

with open(os.path.join(save_dir, "test.txt"), "w") as f:
    f.writelines([f"{img}\n" for img in test_files])

print("Split created successfully!")
print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
