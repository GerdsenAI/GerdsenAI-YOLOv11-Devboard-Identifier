#!/usr/bin/env python3
"""
03_augment_dataset.py - Data augmentation pipeline

This script expands the dataset through various augmentation techniques:
- Geometric: rotation, scale, flip
- Photometric: brightness, contrast, saturation
- Background: synthetic background swapping
- Noise: gaussian noise, blur

Target: 100+ images per class after augmentation

Usage:
    python 03_augment_dataset.py --input ../dataset/raw --output ../dataset/augmented
    python 03_augment_dataset.py --target-per-class 150
"""

import os
import sys
import yaml
import random
import argparse
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default augmentation parameters - constrained to avoid too dark/bright
DEFAULT_CONFIG = {
    'rotation_range': [-12, 12],
    'scale_range': [0.9, 1.1],
    'brightness_range': [0.85, 1.15],  # Reduced from 0.8-1.2
    'contrast_range': [0.9, 1.1],       # Reduced from 0.85-1.15
    'saturation_range': [0.85, 1.15],
    'flip_horizontal': True,
    'flip_vertical': False,
    'gaussian_noise': 0.015,            # Reduced noise
    'blur_range': [0, 0.8],
    'min_brightness': 30,               # Reject images darker than this
    'max_brightness': 240,              # Reject images brighter than this
}

# Background colors for synthetic backgrounds
BACKGROUNDS = [
    (255, 255, 255),  # White
    (240, 240, 240),  # Light gray
    (200, 200, 200),  # Gray
    (50, 50, 50),     # Dark gray
    (0, 0, 0),        # Black
    (245, 245, 220),  # Beige
    (176, 196, 222),  # Light steel blue
    (144, 238, 144),  # Light green
]


class DataAugmenter:
    def __init__(self, config: dict = None):
        self.config = config or DEFAULT_CONFIG

    def _get_edge_color(self, img: Image.Image) -> tuple:
        """Get average color from image edges for fill."""
        arr = np.array(img)
        # Sample from edges
        top = arr[0, :, :].mean(axis=0)
        bottom = arr[-1, :, :].mean(axis=0)
        left = arr[:, 0, :].mean(axis=0)
        right = arr[:, -1, :].mean(axis=0)
        avg = (top + bottom + left + right) / 4
        return tuple(int(c) for c in avg)

    def rotate(self, img: Image.Image) -> Image.Image:
        """Random rotation within range - uses edge color for fill."""
        angle = random.uniform(*self.config['rotation_range'])
        fill_color = self._get_edge_color(img)
        return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=fill_color)

    def scale(self, img: Image.Image) -> Image.Image:
        """Random scaling within range - uses edge color for padding."""
        scale = random.uniform(*self.config['scale_range'])
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)

        scaled = img.resize((new_w, new_h), Image.LANCZOS)

        # Pad or crop to original size - use edge color instead of black
        fill_color = self._get_edge_color(img)
        result = Image.new('RGB', (w, h), fill_color)
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2

        if scale < 1:
            result.paste(scaled, (paste_x, paste_y))
        else:
            crop_x = (new_w - w) // 2
            crop_y = (new_h - h) // 2
            result = scaled.crop((crop_x, crop_y, crop_x + w, crop_y + h))

        return result

    def flip_horizontal(self, img: Image.Image) -> Image.Image:
        """Horizontal flip."""
        if self.config['flip_horizontal'] and random.random() > 0.5:
            return ImageOps.mirror(img)
        return img

    def flip_vertical(self, img: Image.Image) -> Image.Image:
        """Vertical flip."""
        if self.config['flip_vertical'] and random.random() > 0.5:
            return ImageOps.flip(img)
        return img

    def adjust_brightness(self, img: Image.Image) -> Image.Image:
        """Random brightness adjustment."""
        factor = random.uniform(*self.config['brightness_range'])
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)

    def adjust_contrast(self, img: Image.Image) -> Image.Image:
        """Random contrast adjustment."""
        factor = random.uniform(*self.config['contrast_range'])
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)

    def adjust_saturation(self, img: Image.Image) -> Image.Image:
        """Random saturation adjustment."""
        factor = random.uniform(*self.config['saturation_range'])
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)

    def add_noise(self, img: Image.Image) -> Image.Image:
        """Add Gaussian noise."""
        if self.config['gaussian_noise'] > 0:
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, self.config['gaussian_noise'] * 255, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        return img

    def add_blur(self, img: Image.Image) -> Image.Image:
        """Random blur."""
        blur_range = self.config['blur_range']
        if blur_range[1] > 0:
            radius = random.uniform(*blur_range)
            if radius > 0.1:
                return img.filter(ImageFilter.GaussianBlur(radius))
        return img

    def change_background(self, img: Image.Image) -> Image.Image:
        """Attempt to change background color (simple approach)."""
        # This is a simplified version - for production, use proper segmentation
        bg_color = random.choice(BACKGROUNDS)
        # For now, just return original - proper implementation would need segmentation
        return img

    def _validate_brightness(self, img: Image.Image) -> bool:
        """Check if image brightness is within acceptable range."""
        arr = np.array(img)
        avg_brightness = np.mean(arr)
        min_b = self.config.get('min_brightness', 30)
        max_b = self.config.get('max_brightness', 240)
        return min_b <= avg_brightness <= max_b

    def augment_single(self, img: Image.Image, intensity: str = 'medium') -> Image.Image:
        """Apply random augmentations to a single image. Returns None if result is invalid."""
        # Ensure RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Apply augmentations based on intensity
        if intensity == 'light':
            transforms = [
                (0.5, self.flip_horizontal),
                (0.3, self.adjust_brightness),
                (0.3, self.adjust_contrast),
            ]
        elif intensity == 'medium':
            transforms = [
                (0.5, self.flip_horizontal),
                (0.5, self.rotate),
                (0.3, self.scale),
                (0.5, self.adjust_brightness),
                (0.5, self.adjust_contrast),
                (0.3, self.adjust_saturation),
            ]
        else:  # heavy
            transforms = [
                (0.5, self.flip_horizontal),
                (0.6, self.rotate),
                (0.4, self.scale),
                (0.5, self.adjust_brightness),
                (0.5, self.adjust_contrast),
                (0.4, self.adjust_saturation),
                (0.2, self.add_noise),
                (0.2, self.add_blur),
            ]

        result = img.copy()
        for prob, transform in transforms:
            if random.random() < prob:
                try:
                    result = transform(result)
                except Exception as e:
                    logger.debug(f"Transform failed: {e}")

        # Validate result brightness
        if not self._validate_brightness(result):
            return None  # Signal to caller to retry

        return result


def augment_class(args):
    """Augment all images for a single class."""
    input_dir, output_dir, class_name, target_count, config = args

    augmenter = DataAugmenter(config)

    input_path = Path(input_dir) / class_name
    output_path = Path(output_dir) / class_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all images
    images = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png')) + list(input_path.glob('*.jpeg'))

    if not images:
        logger.warning(f"No images found for {class_name}")
        return class_name, 0

    logger.info(f"Processing {class_name}: {len(images)} source images -> target {target_count}")

    # Copy originals first
    count = 0
    for img_path in images:
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            output_file = output_path / f"{class_name}_{count:04d}.jpg"
            img.save(output_file, 'JPEG', quality=95)
            count += 1
        except Exception as e:
            logger.debug(f"Failed to copy {img_path}: {e}")

    # Generate augmented versions
    max_retries = 5  # Prevent infinite loops on problematic images
    while count < target_count:
        # Pick random source image
        src_path = random.choice(images)
        try:
            img = Image.open(src_path)

            # Vary augmentation intensity
            intensity = random.choice(['light', 'medium', 'medium', 'heavy'])

            # Try augmentation with retries if brightness is invalid
            for retry in range(max_retries):
                augmented = augmenter.augment_single(img, intensity)
                if augmented is not None:
                    break
            else:
                # All retries failed, skip this image
                continue

            output_file = output_path / f"{class_name}_{count:04d}.jpg"
            augmented.save(output_file, 'JPEG', quality=95)
            count += 1

        except Exception as e:
            logger.debug(f"Augmentation failed: {e}")

    logger.info(f"Completed {class_name}: {count} images")
    return class_name, count


def main():
    parser = argparse.ArgumentParser(description='Augment dev board dataset')
    parser.add_argument('--input', type=str, default='../dataset/raw',
                        help='Input directory with raw images')
    parser.add_argument('--output', type=str, default='../dataset/augmented',
                        help='Output directory for augmented images')
    parser.add_argument('--config', type=str, default='../config/boards.yaml',
                        help='Path to boards.yaml for augmentation config')
    parser.add_argument('--target-per-class', type=int, default=100,
                        help='Target number of images per class')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input
    output_dir = script_dir / args.output
    config_path = script_dir / args.config

    # Load config for augmentation parameters
    aug_config = DEFAULT_CONFIG
    if config_path.exists():
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            if 'augmentation' in full_config:
                aug_config.update(full_config['augmentation'])

    # Get all class directories
    classes = [d.name for d in input_dir.iterdir() if d.is_dir()]

    if not classes:
        logger.error(f"No class directories found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(classes)} classes")
    logger.info(f"Target per class: {args.target_per_class}")

    # Prepare arguments for parallel processing
    task_args = [
        (str(input_dir), str(output_dir), cls, args.target_per_class, aug_config)
        for cls in classes
    ]

    # Process in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(augment_class, task_args):
            results.append(result)

    # Summary
    print("\n" + "=" * 50)
    print("AUGMENTATION COMPLETE")
    print("=" * 50)
    total = 0
    for class_name, count in results:
        print(f"  {class_name}: {count} images")
        total += count
    print(f"\nTotal: {total} images")
    print(f"Output: {output_dir}")
    print("=" * 50)


if __name__ == '__main__':
    main()
