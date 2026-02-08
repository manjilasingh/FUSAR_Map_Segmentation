import os, random

root = "data/FUSAR-Map/SAR_512"
names = [f for f in os.listdir(root) if f.endswith(".png")]
random.shuffle(names)

train_split = int(0.8 * len(names))
val_split   = int(0.9 * len(names))

with open("data/FUSAR-Map/train.txt", "w") as f:
    f.writelines(n + "\n" for n in names[:train_split])
with open("data/FUSAR-Map/val.txt", "w") as f:
    f.writelines(n + "\n" for n in names[train_split:val_split])
with open("data/FUSAR-Map/test.txt", "w") as f:
    f.writelines(n + "\n" for n in names[val_split:])
