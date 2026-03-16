from pathlib import Path
import numpy as np
from PIL import Image

MASKS_DIR = Path("data/processed/CholecSeg8k/masks")

all_colors = set()

for i, mask_path in enumerate(sorted(MASKS_DIR.glob("*.png"))):
    mask = Image.open(mask_path).convert("RGB")
    mask_np = np.array(mask, dtype=np.uint8)
    unique_colors = np.unique(mask_np.reshape(-1, 3), axis=0)
    for color in unique_colors:
        all_colors.add(tuple(int(x) for x in color))

    if i % 200 == 0:
        print(f"Processed {i} masks...")

all_colors = sorted(all_colors)
print(f"\nTotal unique RGB colors found: {len(all_colors)}")
for color in all_colors:
    print(color)