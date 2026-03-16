from typing import Callable, Tuple

import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_image(image_np: np.ndarray) -> np.ndarray:
    image_np = image_np.astype(np.float32) / 255.0
    image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
    return image_np


class SegmentationTransform:
    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        train: bool = False,
        hflip_prob: float = 0.5,
    ) -> None:
        self.image_size = image_size
        self.train = train
        self.hflip_prob = hflip_prob

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")

        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        image_np = np.array(image, dtype=np.uint8)
        mask_np = np.array(mask, dtype=np.uint8)

        if self.train:
            if np.random.rand() < self.hflip_prob:
                image_np = np.fliplr(image_np).copy()
                mask_np = np.fliplr(mask_np).copy()

            # mild brightness jitter
            if np.random.rand() < 0.5:
                factor = np.random.uniform(0.9, 1.1)
                image_np = np.clip(image_np.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        image_np = normalize_image(image_np)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().contiguous()
        mask_tensor = torch.from_numpy(mask_np).long().contiguous()

        return image_tensor, mask_tensor


def get_train_transform(image_size: Tuple[int, int] = (512, 512)) -> Callable:
    return SegmentationTransform(image_size=image_size, train=True)


def get_val_transform(image_size: Tuple[int, int] = (512, 512)) -> Callable:
    return SegmentationTransform(image_size=image_size, train=False)