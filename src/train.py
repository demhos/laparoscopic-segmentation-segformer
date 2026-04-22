from pathlib import Path
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cholec_dataset import CholecSeg8kDataset, NUM_CLASSES, IGNORE_INDEX
from eval import run_validation
from model_utils import build_segformer
from transforms import get_train_transform, get_val_transform


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(PROJECT_ROOT / "data" / "processed" / "CholecSeg8k")))
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", str(PROJECT_ROOT / "checkpoints")))
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

IMAGES_DIR = DATA_ROOT / "images"
MASKS_DIR = DATA_ROOT / "masks"
SPLITS_DIR = DATA_ROOT / "splits"

TRAIN_SPLIT = SPLITS_DIR / "train.txt"
VAL_SPLIT = SPLITS_DIR / "val.txt"

SEED = 42
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

OVERFIT_TINY_BATCH = False
OVERFIT_SAMPLES = 16


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset):
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for i in range(len(dataset)):
        labels = dataset[i]["labels"].numpy()
        valid = labels != IGNORE_INDEX
        vals = labels[valid]
        if vals.size > 0:
            counts += np.bincount(vals, minlength=NUM_CLASSES)

    freqs = counts / np.maximum(counts.sum(), 1)

    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    positive = counts > 0
    weights[positive] = 1.0 / np.sqrt(freqs[positive] + 1e-8)

    if positive.any():
        weights[positive] = weights[positive] / weights[positive].mean()

    # milder weighting than before
    weights = np.clip(weights, 0.5, 2.0)

    print("Class pixel counts:", counts.tolist())
    print("Class weights:", weights.tolist())

    return torch.tensor(weights, dtype=torch.float32)


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

    class_weights = compute_class_weights(train_dataset)

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

    return train_loader, val_loader, class_weights


def dice_loss(logits, targets, num_classes, ignore_index=255, smooth=1e-5):
    probs = torch.softmax(logits, dim=1)

    valid_mask = (targets != ignore_index).unsqueeze(1)  # [B,1,H,W]
    targets_clean = targets.clone()
    targets_clean[targets_clean == ignore_index] = 0

    one_hot = F.one_hot(targets_clean, num_classes=num_classes).permute(0, 3, 1, 2).float()

    probs = probs * valid_mask
    one_hot = one_hot * valid_mask

    intersection = (probs * one_hot).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def train_one_epoch(model, dataloader, optimizer, device, criterion):
    model.train()

    total_loss = 0.0
    total_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True).contiguous()
        labels = batch["labels"].to(device, non_blocking=True).contiguous()

        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).contiguous()

        ce = criterion(logits, labels)
        d = dice_loss(logits, labels, NUM_CLASSES, ignore_index=IGNORE_INDEX)
        loss = ce + d

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ce=f"{ce.item():.4f}",
            dice=f"{d.item():.4f}",
        )

    return total_loss / max(total_batches, 1)


def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    pin_memory = (device.type == "cuda")

    train_loader, val_loader, class_weights = make_dataloaders(pin_memory)

    model = build_segformer().to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        ignore_index=IGNORE_INDEX,
    )

    best_val_miou = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_metrics = run_validation(model, val_loader, device, criterion)

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