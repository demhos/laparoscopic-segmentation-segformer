from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets import CholecSeg8kDataset
from eval import run_validation
from model_utils import build_segformer
from transforms import get_train_transform, get_val_transform

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data" / "processed" / "CholecSeg8k"))

IMAGES_DIR = DATA_ROOT / "images"
MASKS_DIR = DATA_ROOT / "masks"
SPLITS_DIR = DATA_ROOT / "splits"

TRAIN_SPLIT = SPLITS_DIR / "train.txt"
VAL_SPLIT = SPLITS_DIR / "val.txt"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


SEED = 42
IMAGE_SIZE = (384, 384)
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

# Set this to True for debugging
OVERFIT_TINY_BATCH = False
OVERFIT_SAMPLES = 16

IGNORE_INDEX = 255


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataloaders(pin_memory: bool):
    train_dataset = CholecSeg8kDataset(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        split_file=TRAIN_SPLIT,
        transform=get_train_transform(IMAGE_SIZE),
    )

    val_dataset = CholecSeg8kDataset(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        split_file=VAL_SPLIT,
        transform=get_val_transform(IMAGE_SIZE),
    )

    if OVERFIT_TINY_BATCH:
        tiny_indices = list(range(min(OVERFIT_SAMPLES, len(train_dataset))))
        train_dataset = Subset(train_dataset, tiny_indices)
        val_dataset = Subset(
            val_dataset,
            tiny_indices[: min(len(tiny_indices), len(val_dataset))]
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_batches = 0

    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device).contiguous()
        labels = batch["labels"].to(device).contiguous()

        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).contiguous()

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_batches, 1)


def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    pin_memory = (device.type == "cuda")

    train_loader, val_loader = make_dataloaders(pin_memory)

    model = build_segformer().to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_miou = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = run_validation(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_metrics['loss']:.4f}")
        print(f"Val mIoU:   {val_metrics['miou']:.4f}")

        latest_ckpt = CHECKPOINT_DIR / "segformer_latest.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": val_metrics["miou"],
            },
            latest_ckpt,
        )

        if val_metrics["miou"] > best_val_miou:
            best_val_miou = val_metrics["miou"]
            best_ckpt = CHECKPOINT_DIR / "segformer_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_miou": val_metrics["miou"],
                },
                best_ckpt,
            )
            print(f"Saved new best checkpoint to {best_ckpt}")


if __name__ == "__main__":
    main()