#!/usr/bin/env python3
"""
04_prepare_yolo_dataset.py - Convert dataset to YOLO format

This script:
1. Creates YOLO-format dataset structure
2. Auto-generates bounding box annotations (full image = board)
3. Splits into train/val/test sets
4. Creates data.yaml configuration

For proper bounding boxes, use Roboflow or manual annotation.
This script creates placeholder annotations assuming the board fills most of the image.

Usage:
    python 04_prepare_yolo_dataset.py --input ../dataset/augmented --output ../dataset/yolo
"""

import os
import sys
import yaml
import random
import shutil
import argparse
from pathlib import Path
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_yolo_annotation(img_path: Path, class_id: int, output_path: Path,
                           bbox_margin: float = 0.1):
    """
    Create YOLO format annotation file.

    For this simplified version, we assume the board is centered and fills
    most of the image. For production, use proper annotation tools.

    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1]
    """
    try:
        with Image.open(img_path) as img:
            w, h = img.size

        # Assume board is centered with margin
        # In production, this should come from actual annotations
        x_center = 0.5
        y_center = 0.5
        width = 1.0 - (2 * bbox_margin)
        height = 1.0 - (2 * bbox_margin)

        # Write annotation
        with open(output_path, 'w') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        return True

    except Exception as e:
        logger.debug(f"Failed to create annotation for {img_path}: {e}")
        return False


def prepare_dataset(input_dir: Path, output_dir: Path, config_path: Path,
                    train_ratio: float = 0.7, val_ratio: float = 0.2):
    """Prepare YOLO format dataset."""

    # Load board config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    boards = config.get('boards', {})
    class_names = list(boards.keys())

    logger.info(f"Found {len(class_names)} classes")

    # Create directory structure
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Process each class
    all_stats = {'train': 0, 'val': 0, 'test': 0}

    for class_id, class_name in enumerate(class_names):
        class_dir = input_dir / class_name

        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            continue

        # Get all images
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        random.shuffle(images)

        if not images:
            logger.warning(f"No images for {class_name}")
            continue

        # Split dataset
        n_train = int(len(images) * train_ratio)
        n_val = int(len(images) * val_ratio)

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        for split, split_images in splits.items():
            for img_path in split_images:
                # Copy image
                dest_img = output_dir / 'images' / split / img_path.name
                shutil.copy2(img_path, dest_img)

                # Create annotation
                dest_label = output_dir / 'labels' / split / (img_path.stem + '.txt')
                create_yolo_annotation(img_path, class_id, dest_label)

                all_stats[split] += 1

        logger.info(f"  {class_name}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Create data.yaml
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names)
    }

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    logger.info(f"\nCreated data.yaml: {yaml_path}")

    # Also create classes.txt
    classes_path = output_dir / 'classes.txt'
    with open(classes_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

    return all_stats, class_names


def main():
    parser = argparse.ArgumentParser(description='Prepare YOLO format dataset')
    parser.add_argument('--input', type=str, default='../dataset/augmented',
                        help='Input directory with augmented images')
    parser.add_argument('--output', type=str, default='../dataset/yolo',
                        help='Output directory for YOLO dataset')
    parser.add_argument('--config', type=str, default='../config/boards.yaml',
                        help='Path to boards.yaml')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input
    output_dir = script_dir / args.output
    config_path = script_dir / args.config

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    stats, classes = prepare_dataset(
        input_dir, output_dir, config_path,
        args.train_ratio, args.val_ratio
    )

    print("\n" + "=" * 50)
    print("YOLO DATASET PREPARED")
    print("=" * 50)
    print(f"Classes: {len(classes)}")
    print(f"  Train: {stats['train']} images")
    print(f"  Val:   {stats['val']} images")
    print(f"  Test:  {stats['test']} images")
    print(f"\nOutput: {output_dir}")
    print(f"Config: {output_dir / 'data.yaml'}")
    print("=" * 50)
    print("\nNOTE: This script creates approximate bounding boxes.")
    print("For production, annotate properly using Roboflow or LabelImg.")


if __name__ == '__main__':
    main()
