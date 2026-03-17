# Laparoscopic Surgical Segmentation with SegFormer

Semantic segmentation of laparoscopic surgery frames using the CholecSeg8k dataset and SegFormer.

## Dataset
CholecSeg8k

## Model
SegFormer-B0

## Pipeline

1. Dataset restructuring
2. Mask remapping
3. SegFormer fine-tuning
4. mIoU evaluation

## Training

```bash
python src/train.py