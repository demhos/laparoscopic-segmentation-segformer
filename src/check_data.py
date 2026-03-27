from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from cholec_dataset import CholecSeg8kDataset
from transforms import get_val_transform


import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data" / "processed" / "CholecSeg8k"))

IMAGES_DIR = DATA_ROOT / "images"
MASKS_DIR = DATA_ROOT / "masks"
SPLITS_DIR = DATA_ROOT / "splits"

TRAIN_SPLIT = SPLITS_DIR / "train.txt"
VAL_SPLIT = SPLITS_DIR / "val.txt"
TEST_SPLIT = SPLITS_DIR / "test.txt"


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    palette = np.array([
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [255, 128, 0],
    ], dtype=np.uint8)

    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask != 255
    color_mask[valid] = palette[mask[valid] % len(palette)]
    return color_mask


def denormalize_image(image: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image * std) + mean
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)


def overlay_image_and_mask(image: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image = image.astype(np.float32)
    mask_rgb = mask_rgb.astype(np.float32)
    overlay = (1 - alpha) * image + alpha * mask_rgb
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_sample_visualizations(dataset: CholecSeg8kDataset, out_dir: Path, num_samples: int = 5) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(num_samples, len(dataset))):
        item = dataset[i]
        image = item["pixel_values"].permute(1, 2, 0).numpy()
        image = denormalize_image(image)
        mask = item["labels"].numpy()

        mask_rgb = colorize_mask(mask)
        overlay = overlay_image_and_mask(image, mask_rgb)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title(f"Image: {item['id']}")
        axes[0].axis("off")

        axes[1].imshow(mask_rgb)
        axes[1].set_title("Mask")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(out_dir / f"{item['id']}_viz.png", dpi=150)
        plt.close(fig)


def main() -> None:
    transform = get_val_transform((512, 512))

    train_dataset = CholecSeg8kDataset(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        split_file=TRAIN_SPLIT,
        transform=transform,
    )

    val_dataset = CholecSeg8kDataset(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        split_file=VAL_SPLIT,
        transform=transform,
    )

    test_dataset = CholecSeg8kDataset(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        split_file=TEST_SPLIT,
        transform=transform,
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size:   {len(val_dataset)}")
    print(f"Test size:  {len(test_dataset)}")

    sample = train_dataset[0]
    print("\nSample keys:", sample.keys())
    print("Image shape:", sample["pixel_values"].shape)
    print("Mask shape: ", sample["labels"].shape)
    print("Sample ID:  ", sample["id"])
    print("Unique mask values:", torch.unique(sample["labels"]).tolist())

    loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    print("\nBatch image shape:", batch["pixel_values"].shape)
    print("Batch mask shape: ", batch["labels"].shape)

    viz_dir = PROJECT_ROOT / "outputs" / "phase_b_checks"
    save_sample_visualizations(train_dataset, viz_dir, num_samples=5)
    print(f"\nSaved sample visualizations to: {viz_dir}")


if __name__ == "__main__":
    main()