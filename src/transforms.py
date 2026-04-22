from typing import Callable, Tuple
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance


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
        hflip_prob: float = 0.3,
        brightness_prob: float = 0.3,
        contrast_prob: float = 0.3,
        rotate_prob: float = 0.2,
        max_rotate_deg: float = 5.0,
    ) -> None:
        self.image_size = image_size
        self.train = train
        self.hflip_prob = hflip_prob
        self.brightness_prob = brightness_prob
        self.contrast_prob = contrast_prob
        self.rotate_prob = rotate_prob
        self.max_rotate_deg = max_rotate_deg

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")

        if self.train:
            if random.random() < self.hflip_prob:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            if random.random() < self.brightness_prob:
                factor = random.uniform(0.85, 1.15)
                image = ImageEnhance.Brightness(image).enhance(factor)

            if random.random() < self.contrast_prob:
                factor = random.uniform(0.85, 1.15)
                image = ImageEnhance.Contrast(image).enhance(factor)

            if random.random() < self.rotate_prob:
                angle = random.uniform(-self.max_rotate_deg, self.max_rotate_deg)
                image = image.rotate(
                    angle,
                    resample=Image.BILINEAR,
                    fillcolor=(0, 0, 0),
                )
                mask = mask.rotate(
                    angle,
                    resample=Image.NEAREST,
                    fillcolor=255,  # ignore index
                )

        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        image_np = np.array(image, dtype=np.uint8)
        mask_np = np.array(mask, dtype=np.uint8)

        image_np = normalize_image(image_np)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().contiguous()
        mask_tensor = torch.from_numpy(mask_np).long().contiguous()

        return image_tensor, mask_tensor


def get_train_transform(image_size: Tuple[int, int] = (512, 512)) -> Callable:
    return SegmentationTransform(image_size=image_size, train=True)


def get_val_transform(image_size: Tuple[int, int] = (512, 512)) -> Callable:
    return SegmentationTransform(image_size=image_size, train=False)