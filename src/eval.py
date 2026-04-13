import torch
import torch.nn.functional as F

from cholec_dataset import IGNORE_INDEX, NUM_CLASSES


def compute_batch_miou(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> float:
    preds = torch.argmax(logits, dim=1)

    intersection = torch.zeros(num_classes, device=logits.device, dtype=torch.float64)
    union = torch.zeros(num_classes, device=logits.device, dtype=torch.float64)

    valid_mask = labels != ignore_index

    for cls in range(num_classes):
        pred_mask = (preds == cls) & valid_mask
        label_mask = (labels == cls) & valid_mask

        inter = (pred_mask & label_mask).sum()
        uni = (pred_mask | label_mask).sum()

        intersection[cls] += inter
        union[cls] += uni

    iou = intersection / torch.clamp(union, min=1.0)
    valid_classes = union > 0

    if valid_classes.sum() == 0:
        return 0.0

    return iou[valid_classes].mean().item()


@torch.no_grad()
def run_validation(model, dataloader, device, criterion):
    model.eval()

    total_loss = 0.0
    total_miou = 0.0
    total_batches = 0

    for batch in dataloader:
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
        miou = compute_batch_miou(logits, labels)

        total_loss += loss.item()
        total_miou += miou
        total_batches += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "miou": total_miou / max(total_batches, 1),
    }