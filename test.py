import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Load all labels
root = "data/FUSAR-Map/Labels_1024"
lbl_list = sorted(os.listdir(root))

counts = np.zeros(5)

for name in lbl_list:
    mask = np.array(Image.open(os.path.join(root, name)))
    for c in range(5):
        counts[c] += np.sum(mask == c)

classes = ["Water","Road","Building","Vegetation","Others"]

plt.bar(classes, counts)
plt.xticks(rotation=30)
plt.ylabel("Pixel Count")
plt.title("Class Pixel Distribution (Imbalance Visualization)")
plt.show()
