from transformers import SegformerForSemanticSegmentation

from cholec_dataset import IGNORE_INDEX, NUM_CLASSES, CLASS_NAMES


ID2LABEL = {i: name for i, name in enumerate(CLASS_NAMES)}
LABEL2ID = {name: i for i, name in ID2LABEL.items()}


def build_segformer(model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512"):
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    model.config.num_labels = NUM_CLASSES
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID
    model.config.semantic_loss_ignore_index = IGNORE_INDEX

    return model