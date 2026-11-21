"""
Dataset Converter & Preprocessor:
Converts Ambulance.v1i.tensorflow (TensorFlow CSV format) into standard YOLOv8 Darknet format.
Splits images into train (85%) and val (15%) partitions and generates data.yaml.
Usage: python scripts/prepare_ambulance_dataset.py
"""
import os
import shutil
import random
import yaml
import pandas as pd
from pathlib import Path


CLASS_MAPPING = {
    "Ambulance": 0,
    "ambulance": 0,
    "Fire truck": 1,
    "fire truck": 1,
    "Fire Truck": 1
}


def prepare_yolo_dataset(
    source_dir: str = "data/datasets/Ambulance.v1i.tensorflow",
    target_dir: str = "data/datasets/ambulance_yolo",
    val_ratio: float = 0.15,
    seed: int = 42
):
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Check primary or fallback root location
    if not (source_path / "train" / "_annotations.csv").exists():
        fallback = Path("Ambulance.v1i.tensorflow")
        if (fallback / "train" / "_annotations.csv").exists():
            source_path = fallback

    csv_path = source_path / "train" / "_annotations.csv"
    train_images_source = source_path / "train"

    if not csv_path.exists():
        raise FileNotFoundError(f"Annotation file not found at: {csv_path}")

    print("=" * 70)
    print("      AMBULANCE DATASET CONVERTER TO YOLOV8 FORMAT")
    print("=" * 70)
    print(f" Source CSV:       {csv_path}")
    print(f" Target Directory: {target_path}")

    # Read annotations
    df = pd.read_csv(csv_path)
    total_annotations = len(df)
    unique_images = df["filename"].unique()
    print(f" Found {total_annotations} bounding box annotations across {len(unique_images)} images.")

    # Random split with fixed seed
    random.seed(seed)
    image_list = list(unique_images)
    random.shuffle(image_list)

    val_count = int(len(image_list) * val_ratio)
    val_files = set(image_list[:val_count])
    train_files = set(image_list[val_count:])

    print(f" Split summary: {len(train_files)} Train images, {len(val_files)} Validation images.")

    # Setup target directories
    for split in ["train", "val"]:
        (target_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (target_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Group annotations by image filename
    grouped = df.groupby("filename")

    copied_images = 0
    written_labels = 0

    for filename, group in grouped:
        src_img = train_images_source / filename
        if not src_img.exists():
            print(f"Warning: Missing image file {src_img}, skipping.")
            continue

        split = "val" if filename in val_files else "train"
        dst_img = target_path / "images" / split / filename
        txt_name = Path(filename).stem + ".txt"
        dst_label = target_path / "labels" / split / txt_name

        # Copy image if not already present
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
            copied_images += 1

        # Build YOLO formatted labels
        lines = []
        for _, row in group.iterrows():
            cls_name = row["class"]
            cls_id = CLASS_MAPPING.get(cls_name, 0)
            img_w = float(row["width"])
            img_h = float(row["height"])
            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            xmax = float(row["xmax"])
            ymax = float(row["ymax"])

            # Compute YOLO normalized bounding box coordinates
            box_w = max(0.001, (xmax - xmin) / img_w)
            box_h = max(0.001, (ymax - ymin) / img_h)
            x_center = ((xmin + xmax) / 2.0) / img_w
            y_center = ((ymin + ymax) / 2.0) / img_h

            # Clamp coordinates strictly between 0.0 and 1.0
            x_center = min(1.0, max(0.0, x_center))
            y_center = min(1.0, max(0.0, y_center))
            box_w = min(1.0, max(0.001, box_w))
            box_h = min(1.0, max(0.001, box_h))

            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        with open(dst_label, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        written_labels += 1

    # Generate data.yaml
    data_yaml = {
        "path": str(target_path.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "ambulance",
            1: "fire_truck"
        }
    }

    yaml_dest = target_path / "data.yaml"
    with open(yaml_dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f" Generated data.yaml at {yaml_dest}")
    print(f" Converted successfully: {copied_images} images copied, {written_labels} label files created.")
    print("=" * 70)
    return yaml_dest


if __name__ == "__main__":
    prepare_yolo_dataset()
