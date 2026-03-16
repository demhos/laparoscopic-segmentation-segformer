from pathlib import Path
import shutil
import re


RAW_ROOT = Path("data/raw/CholecSeg8k")
PROCESSED_ROOT = Path("data/processed/CholecSeg8k")

IMAGES_OUT = PROCESSED_ROOT / "images"
MASKS_OUT = PROCESSED_ROOT / "masks"
SPLITS_OUT = PROCESSED_ROOT / "splits"


def ensure_dirs() -> None:
    IMAGES_OUT.mkdir(parents=True, exist_ok=True)
    MASKS_OUT.mkdir(parents=True, exist_ok=True)
    SPLITS_OUT.mkdir(parents=True, exist_ok=True)


def is_video_dir(path: Path) -> bool:
    return path.is_dir() and re.fullmatch(r"video\d+", path.name) is not None


def is_clip_dir(path: Path) -> bool:
    return path.is_dir() and re.fullmatch(r"video\d+_\d+", path.name) is not None


def parse_frame_number(filename: str) -> str | None:
    match = re.fullmatch(r"frame_(\d+)_endo(?:_watershed_mask)?\.png", filename)
    if match:
        return match.group(1)
    return None


def collect_pairs_from_clip(video_dir: Path, clip_dir: Path) -> list[tuple[Path, Path, str]]:
    pairs = []

    png_files = list(clip_dir.glob("*.png"))

    image_files = {}
    mask_files = {}

    for file_path in png_files:
        name = file_path.name

        if name.endswith("_color_mask.png"):
            continue
        if name.endswith("_endo_mask.png"):
            continue

        if name.endswith("_endo_watershed_mask.png"):
            frame_num = parse_frame_number(name)
            if frame_num is not None:
                mask_files[frame_num] = file_path
            continue

        if name.endswith("_endo.png"):
            frame_num = parse_frame_number(name)
            if frame_num is not None:
                image_files[frame_num] = file_path
            continue

    common_frames = sorted(set(image_files.keys()) & set(mask_files.keys()), key=lambda x: int(x))

    for frame_num in common_frames:
        image_path = image_files[frame_num]
        mask_path = mask_files[frame_num]
        new_base_name = f"{video_dir.name}_{clip_dir.name.split('_')[1]}_frame_{frame_num}"
        pairs.append((image_path, mask_path, new_base_name))

    missing_images = sorted(set(mask_files.keys()) - set(image_files.keys()), key=lambda x: int(x))
    missing_masks = sorted(set(image_files.keys()) - set(mask_files.keys()), key=lambda x: int(x))

    if missing_images:
        print(f"[WARN] Missing images in {clip_dir}: frames {missing_images[:10]}")
    if missing_masks:
        print(f"[WARN] Missing masks in {clip_dir}: frames {missing_masks[:10]}")

    return pairs


def copy_pairs(pairs: list[tuple[Path, Path, str]]) -> int:
    copied = 0

    for image_path, mask_path, new_base_name in pairs:
        out_image = IMAGES_OUT / f"{new_base_name}.png"
        out_mask = MASKS_OUT / f"{new_base_name}.png"

        if out_image.exists() or out_mask.exists():
            raise FileExistsError(
                f"Output file already exists:\n"
                f"  {out_image}\n"
                f"  {out_mask}\n"
                f"Delete processed folder first if you want a clean rebuild."
            )

        shutil.copy2(image_path, out_image)
        shutil.copy2(mask_path, out_mask)
        copied += 1

    return copied


def create_video_level_splits() -> None:
    train_videos = [f"video{i:02d}" for i in range(1, 12)]
    val_videos = [f"video{i:02d}" for i in range(12, 15)]
    test_videos = [f"video{i:02d}" for i in range(15, 18)]

    all_ids = [p.stem for p in IMAGES_OUT.glob("*.png")]

    train_ids = sorted([x for x in all_ids if any(x.startswith(v + "_") for v in train_videos)])
    val_ids = sorted([x for x in all_ids if any(x.startswith(v + "_") for v in val_videos)])
    test_ids = sorted([x for x in all_ids if any(x.startswith(v + "_") for v in test_videos)])

    (SPLITS_OUT / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (SPLITS_OUT / "val.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    (SPLITS_OUT / "test.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")

    print(f"[INFO] train split: {len(train_ids)}")
    print(f"[INFO] val split:   {len(val_ids)}")
    print(f"[INFO] test split:  {len(test_ids)}")


def main() -> None:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"Raw dataset folder not found: {RAW_ROOT}")

    ensure_dirs()

    total_pairs = 0
    total_clips = 0
    total_videos = 0

    video_dirs = sorted([p for p in RAW_ROOT.iterdir() if is_video_dir(p)])

    if not video_dirs:
        raise RuntimeError(f"No video directories found under: {RAW_ROOT}")

    for video_dir in video_dirs:
        total_videos += 1
        clip_dirs = sorted([p for p in video_dir.iterdir() if is_clip_dir(p)])

        if not clip_dirs:
            print(f"[WARN] No clip folders found in {video_dir}")
            continue

        for clip_dir in clip_dirs:
            total_clips += 1
            pairs = collect_pairs_from_clip(video_dir, clip_dir)
            total_pairs += copy_pairs(pairs)

    print(f"[INFO] videos processed: {total_videos}")
    print(f"[INFO] clips processed:  {total_clips}")
    print(f"[INFO] pairs copied:     {total_pairs}")

    create_video_level_splits()

    image_count = len(list(IMAGES_OUT.glob("*.png")))
    mask_count = len(list(MASKS_OUT.glob("*.png")))

    print(f"[INFO] final image count: {image_count}")
    print(f"[INFO] final mask count:  {mask_count}")

    if image_count != mask_count:
        print("[WARN] Image and mask counts do not match.")

    if image_count != 8080:
        print("[WARN] Expected about 8080 pairs for CholecSeg8k. Check your raw dataset and logs.")
    else:
        print("[INFO] Dataset restructuring looks correct.")


if __name__ == "__main__":
    main()