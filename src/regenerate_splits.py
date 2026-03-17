from pathlib import Path
import re
from collections import Counter

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data" / "processed" / "CholecSeg8k"))
IMAGES_DIR = DATA_ROOT / "images"
SPLITS_DIR = DATA_ROOT / "splits"

print("Using images dir:", IMAGES_DIR)
print("Images dir exists:", IMAGES_DIR.exists())

all_pngs = sorted(IMAGES_DIR.glob("*.png"))
print("Found image files:", len(all_pngs))

SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def extract_video_name(stem: str) -> str:
    match = re.match(r"(video\d+)_", stem)
    if not match:
        raise ValueError(f"Could not extract video name from filename: {stem}")
    return match.group(1)


all_ids = [p.stem for p in all_pngs]

video_counts = Counter()
for stem in all_ids:
    video_name = extract_video_name(stem)
    video_counts[video_name] += 1

print("\nCounts per video:")
for video_name in sorted(video_counts):
    print(f"{video_name}: {video_counts[video_name]}")

train_videos = {
    "video01", "video09", "video12", "video17", "video18", "video20",
    "video24", "video25", "video26", "video27", "video28"
}

val_videos = {
    "video35", "video37", "video43"
}

test_videos = {
    "video48", "video52", "video55"
}

all_split_videos = train_videos | val_videos | test_videos
all_dataset_videos = set(video_counts.keys())

if all_split_videos != all_dataset_videos:
    print("\n[WARN] Split video IDs do not exactly match dataset video IDs.")
    print("In dataset but not split:", sorted(all_dataset_videos - all_split_videos))
    print("In split but not dataset:", sorted(all_split_videos - all_dataset_videos))

train_ids = [x for x in all_ids if extract_video_name(x) in train_videos]
val_ids = [x for x in all_ids if extract_video_name(x) in val_videos]
test_ids = [x for x in all_ids if extract_video_name(x) in test_videos]

(SPLITS_DIR / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
(SPLITS_DIR / "val.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")
(SPLITS_DIR / "test.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")

print("\ntrain:", len(train_ids))
print("val:  ", len(val_ids))
print("test: ", len(test_ids))
print("total:", len(train_ids) + len(val_ids) + len(test_ids))