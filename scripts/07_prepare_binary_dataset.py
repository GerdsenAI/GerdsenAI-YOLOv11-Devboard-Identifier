#!/usr/bin/env python3
"""
07_prepare_binary_dataset.py - Prepare binary classification dataset
Consolidates all images into teensy_41 vs not_teensy classes
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_DIR = PROJECT_DIR / "dataset" / "raw"
AUG_DIR = PROJECT_DIR / "dataset" / "augmented_binary"
YOLO_DIR = PROJECT_DIR / "dataset" / "yolo_binary"

TARGET_PER_CLASS = 200

def collect_images():
    """Collect all images into binary classes"""
    teensy_images = []
    not_teensy_images = []
    
    for class_dir in RAW_DIR.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        if class_name == "teensy_41":
            teensy_images.extend(images)
            print(f"  teensy_41: {len(images)} images")
        elif class_name != "attribution.json":
            not_teensy_images.extend(images)
            print(f"  {class_name}: {len(images)} -> not_teensy")
    
    print(f"\nTotal: teensy_41={len(teensy_images)}, not_teensy={len(not_teensy_images)}")
    return teensy_images, not_teensy_images


def augment_image(img):
    """Apply random augmentation"""
    img_array = np.array(img)
    
    # Random rotation (-15 to 15 degrees)
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, fillcolor=(128, 128, 128), expand=False)
    
    # Random brightness (0.8 to 1.2)
    brightness = random.uniform(0.85, 1.15)
    img_array = np.array(img)
    img_array = np.clip(img_array * brightness, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # Random horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    return img


def augment_to_target(images, class_name, target_count):
    """Augment images to reach target count"""
    output_dir = AUG_DIR / class_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    
    # First, copy originals
    for img_path in images:
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((640, 640))
            img.save(output_dir / f"{class_name}_{count:04d}.jpg")
            count += 1
        except Exception as e:
            print(f"Error with {img_path}: {e}")
    
    print(f"  Copied {count} originals for {class_name}")
    
    # Augment until we reach target
    while count < target_count:
        src_img = random.choice(images)
        try:
            img = Image.open(src_img).convert('RGB')
            img = img.resize((640, 640))
            img = augment_image(img)
            img.save(output_dir / f"{class_name}_{count:04d}.jpg")
            count += 1
        except Exception as e:
            continue
    
    print(f"  Total {count} images for {class_name}")
    return count


def create_yolo_dataset():
    """Create YOLO format dataset with train/val/test splits"""
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    
    classes = ["teensy_41", "not_teensy"]
    
    for split in ["train", "val", "test"]:
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    for class_id, class_name in enumerate(classes):
        aug_dir = AUG_DIR / class_name
        images = list(aug_dir.glob("*.jpg"))
        random.shuffle(images)
        
        n = len(images)
        train_end = int(n * 0.7)
        val_end = int(n * 0.9)
        
        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }
        
        for split_name, split_images in splits.items():
            for img_path in split_images:
                # Copy image
                dest_img = YOLO_DIR / "images" / split_name / img_path.name
                shutil.copy(img_path, dest_img)
                
                # Create label (full image bounding box)
                label_path = YOLO_DIR / "labels" / split_name / img_path.with_suffix('.txt').name
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 0.8 0.8\n")
        
        print(f"  {class_name}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    
    # Create data.yaml
    data_yaml = YOLO_DIR / "data.yaml"
    with open(data_yaml, 'w') as f:
        f.write(f"path: {YOLO_DIR.absolute()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("\n")
        f.write("nc: 2\n")
        f.write("names:\n")
        f.write("  0: teensy_41\n")
        f.write("  1: not_teensy\n")
    
    print(f"\nCreated data.yaml at {data_yaml}")


def main():
    print("=" * 50)
    print("BINARY DATASET PREPARATION")
    print("=" * 50)
    
    # Clean output dirs
    if AUG_DIR.exists():
        shutil.rmtree(AUG_DIR)
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    
    print("\n1. Collecting images...")
    teensy_images, not_teensy_images = collect_images()
    
    print(f"\n2. Augmenting to {TARGET_PER_CLASS} per class...")
    augment_to_target(teensy_images, "teensy_41", TARGET_PER_CLASS)
    augment_to_target(not_teensy_images, "not_teensy", TARGET_PER_CLASS)
    
    print("\n3. Creating YOLO dataset...")
    create_yolo_dataset()
    
    print("\n" + "=" * 50)
    print("DATASET READY!")
    print("=" * 50)


if __name__ == "__main__":
    main()
