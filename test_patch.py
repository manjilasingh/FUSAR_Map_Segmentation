import numpy as np
from PIL import Image, ImageDraw

PATCH_SIZE = 512
STRIDE = 256
IMAGE_PATH = "data/FUSAR-Map/SAR_1024/JiuJiang_01_01_01_01_SAR_001.tif"


OUT_PATH = "horizontal_overlap_hatched_only_overlap.png"

def draw_dashed_rectangle(draw, box, color, width=3, dash=15, gap=10):
    x1, y1, x2, y2 = box

    # top edge
    x = x1
    while x < x2:
        draw.line([(x, y1), (min(x + dash, x2), y1)], fill=color, width=width)
        x += dash + gap

    # bottom edge
    x = x1
    while x < x2:
        draw.line([(x, y2), (min(x + dash, x2), y2)], fill=color, width=width)
        x += dash + gap

    # left edge
    y = y1
    while y < y2:
        draw.line([(x1, y), (x1, min(y + dash, y2))], fill=color, width=width)
        y += dash + gap

    # right edge
    y = y1
    while y < y2:
        draw.line([(x2, y), (x2, min(y + dash, y2))], fill=color, width=width)
        y += dash + gap


def draw_hatched_overlap(draw, box, spacing=10, width=2, color=(0,0,0)):
    x1, y1, x2, y2 = box

    # Draw diagonal hatch lines inside the overlap
    for offset in range(-PATCH_SIZE, PATCH_SIZE*2, spacing):
        draw.line(
            [(x1 + offset, y1), (x1 + offset + PATCH_SIZE, y1 + PATCH_SIZE)],
            fill=color,
            width=width
        )


def draw_horizontal_overlap_hatched(image_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Patch 1 — solid outline
    patch1 = (0, 0, PATCH_SIZE, PATCH_SIZE)
    draw.rectangle(patch1, outline="red", width=4)

    # Patch 2 — dashed outline
    patch2 = (STRIDE, 0, STRIDE + PATCH_SIZE, PATCH_SIZE)
    draw_dashed_rectangle(draw, patch2, color="blue", width=4)

    # Overlap region (width = 256 px)
    overlap = (
        PATCH_SIZE - STRIDE,   # 256
        0,
        PATCH_SIZE,            # 512
        PATCH_SIZE
    )

    # Shade ONLY the overlap region
    draw_hatched_overlap(draw, overlap, spacing=12, width=2)

    img.save(OUT_PATH)
    print(f"Saved overlap-hatched visualization: {OUT_PATH}")


draw_horizontal_overlap_hatched(IMAGE_PATH)
