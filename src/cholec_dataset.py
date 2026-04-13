from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


RAW_TO_CONTIGUOUS = {
    0: 0,     # Black Background
    5: 1,     # Abdominal Wall
    11: 2,    # Liver
    12: 3,    # Gastrointestinal Tract
    13: 4,    # Fat
    21: 5,    # Grasper
    22: 6,    # Connective Tissue
    23: 7,    # Blood
    24: 8,    # Cystic Duct
    25: 9,    # L-hook Electrocautery
    31: 10,   # Gallbladder
    32: 11,   # Hepatic Vein
    33: 12,   # Liver Ligament
    50: 255,  # ignore / artifact
    255: 255, # ignore index
}

IGNORE_INDEX = 255
NUM_CLASSES = 13

CLASS_NAMES = [
    "Black Background",
    "Abdominal Wall",
    "Liver",
    "Gastrointestinal Tract",
    "Fat",
    "Grasper",
    "Connective Tissue",
    "Blood",
    "Cystic Duct",
    "L-hook Electrocautery",
    "Gallbladder",
    "Hepatic Vein",
    "Liver Ligament",
]


def load_split_file(split_file: str | Path) -> list[str]:
    split_file = Path(split_file)
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with split_file.open("r", encoding="utf-8") as f:
        sample_ids = [line.strip() for line in f if line.strip()]

    if not sample_ids:
        raise ValueError(f"No sample IDs found in split file: {split_file}")

    return sample_ids


def remap_mask(mask: Image.Image) -> np.ndarray:
    mask_np = np.array(mask, dtype=np.uint8)

    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]

    if mask_np.ndim != 2:
        raise ValueError(f"Expected mask shape [H, W] or [H, W, 3], got {mask_np.shape}")

    remapped = np.full(mask_np.shape, fill_value=IGNORE_INDEX, dtype=np.uint8)

    unique_vals = np.unique(mask_np)
    for raw_val in unique_vals:
        if int(raw_val) not in RAW_TO_CONTIGUOUS:
            raise ValueError(
                f"Found unknown raw mask value {int(raw_val)}. "
                f"Known values: {sorted(RAW_TO_CONTIGUOUS.keys())}"
            )
        remapped[mask_np == raw_val] = RAW_TO_CONTIGUOUS[int(raw_val)]

    return remapped


class CholecSeg8kDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        split_file: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.sample_ids = load_split_file(split_file)
        self.transform = transform

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        self._verify_files_exist()

    def _verify_files_exist(self) -> None:
        missing_images = []
        missing_masks = []

        for sample_id in self.sample_ids:
            image_path = self.images_dir / f"{sample_id}.png"
            mask_path = self.masks_dir / f"{sample_id}.png"

            if not image_path.exists():
                missing_images.append(str(image_path))
            if not mask_path.exists():
                missing_masks.append(str(mask_path))

        if missing_images or missing_masks:
            msg = []
            if missing_images:
                msg.append(f"Missing images: {missing_images[:10]}")
            if missing_masks:
                msg.append(f"Missing masks: {missing_masks[:10]}")
            raise FileNotFoundError("\n".join(msg))

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample_id = self.sample_ids[idx]

        image_path = self.images_dir / f"{sample_id}.png"
        mask_path = self.masks_dir / f"{sample_id}.png"

        image = Image.open(image_path).convert("RGB")
        raw_mask = Image.open(mask_path)
        remapped_mask_np = remap_mask(raw_mask)
        remapped_mask = Image.fromarray(remapped_mask_np)

        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image, remapped_mask)
        else:
            image_np = np.array(image, dtype=np.float32) / 255.0
            mask_np = remapped_mask_np.astype(np.int64)

            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
            mask_tensor = torch.from_numpy(mask_np).long().contiguous()

        return {
            "pixel_values": image_tensor,
            "labels": mask_tensor,
            "id": sample_id,
        }