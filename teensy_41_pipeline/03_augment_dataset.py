#!/usr/bin/env python3
"""
03_augment_dataset.py - Teensy 4.1 Data Augmentation

Expands the Teensy 4.1 dataset through augmentation:
- Rotation (with edge-color fill)
- Brightness/Contrast adjustment
- Horizontal flip
- Scale variations
- Gaussian noise and blur

Includes brightness validation to prevent dark/washed-out images.

Usage:
    python 03_augment_dataset.py
    python 03_augment_dataset.py --target 300
"""

import os
import sys
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Augmentation config - tuned to avoid dark/bright artifacts
CONFIG = {
    'rotation_range': [-12, 12],
    'scale_range': [0.9, 1.1],
    'brightness_range': [0.85, 1.15],
    'contrast_range': [0.9, 1.1],
    'saturation_range': [0.85, 1.15],
    'flip_horizontal': True,
    'gaussian_noise': 0.015,
    'blur_range': [0, 0.8],
    'min_brightness': 30,   # Reject if avg < 30
    'max_brightness': 240,  # Reject if avg > 240
}

INPUT_DIR = Path(__file__).parent / 'data' / 'raw'
OUTPUT_DIR = Path(__file__).parent / 'data' / 'augmented'


class TeensyAugmenter:
    def __init__(self, config: dict = None):
        self.config = config or CONFIG

    def _get_edge_color(self, img: Image.Image) -> tuple:
        """Get average color from image edges for fill."""
        arr = np.array(img)
        top = arr[0, :, :].mean(axis=0)
        bottom = arr[-1, :, :].mean(axis=0)
        left = arr[:, 0, :].mean(axis=0)
        right = arr[:, -1, :].mean(axis=0)
        avg = (top + bottom + left + right) / 4
        return tuple(int(c) for c in avg)

    def _validate_brightness(self, img: Image.Image) -> bool:
        """Check if image brightness is acceptable."""
        arr = np.array(img)
        avg = np.mean(arr)
        return self.config['min_brightness'] <= avg <= self.config['max_brightness']

    def rotate(self, img: Image.Image) -> Image.Image:
        angle = random.uniform(*self.config['rotation_range'])
        fill = self._get_edge_color(img)
        return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=fill)

    def scale(self, img: Image.Image) -> Image.Image:
        scale = random.uniform(*self.config['scale_range'])
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        scaled = img.resize((new_w, new_h), Image.LANCZOS)

        fill = self._get_edge_color(img)
        result = Image.new('RGB', (w, h), fill)
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
        if random.random() > 0.5:
            return ImageOps.mirror(img)
        return img

    def adjust_brightness(self, img: Image.Image) -> Image.Image:
        factor = random.uniform(*self.config['brightness_range'])
        return ImageEnhance.Brightness(img).enhance(factor)

    def adjust_contrast(self, img: Image.Image) -> Image.Image:
        factor = random.uniform(*self.config['contrast_range'])
        return ImageEnhance.Contrast(img).enhance(factor)

    def adjust_saturation(self, img: Image.Image) -> Image.Image:
        factor = random.uniform(*self.config['saturation_range'])
        return ImageEnhance.Color(img).enhance(factor)

    def add_noise(self, img: Image.Image) -> Image.Image:
        if self.config['gaussian_noise'] > 0:
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, self.config['gaussian_noise'] * 255, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        return img

    def add_blur(self, img: Image.Image) -> Image.Image:
        radius = random.uniform(*self.config['blur_range'])
        if radius > 0.1:
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img

    def augment(self, img: Image.Image, intensity: str = 'medium') -> Image.Image:
        """Apply augmentations. Returns None if result is invalid."""
        if img.mode != 'RGB':
            img = img.convert('RGB')

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

        # Validate brightness
        if not self._validate_brightness(result):
            return None

        return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Augment Teensy 4.1 dataset')
    parser.add_argument('--input', type=str, default=None, help='Input directory')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--target', type=int, default=200, help='Target image count')
    args = parser.parse_args()

    input_dir = Path(args.input) if args.input else INPUT_DIR
    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get source images
    images = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.jpeg')) + list(input_dir.glob('*.png'))

    if not images:
        logger.error(f"No images found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(images)} source images")
    logger.info(f"Target: {args.target} augmented images")

    augmenter = TeensyAugmenter()
    count = 0

    # Copy originals first
    for img_path in images:
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            output_file = output_dir / f"teensy_41_{count:04d}.jpg"
            img.save(output_file, 'JPEG', quality=95)
            count += 1
        except Exception as e:
            logger.warning(f"Failed to copy {img_path.name}: {e}")

    logger.info(f"Copied {count} original images")

    # Generate augmented versions
    max_retries = 5
    while count < args.target:
        src_path = random.choice(images)
        try:
            img = Image.open(src_path)
            intensity = random.choice(['light', 'medium', 'medium', 'heavy'])

            # Retry if brightness validation fails
            for _ in range(max_retries):
                augmented = augmenter.augment(img, intensity)
                if augmented is not None:
                    break
            else:
                continue

            output_file = output_dir / f"teensy_41_{count:04d}.jpg"
            augmented.save(output_file, 'JPEG', quality=95)
            count += 1

            if count % 50 == 0:
                logger.info(f"Progress: {count}/{args.target}")

        except Exception as e:
            logger.debug(f"Augmentation failed: {e}")

    print("\n" + "=" * 50)
    print("AUGMENTATION COMPLETE")
    print("=" * 50)
    print(f"Generated: {count} images")
    print(f"Output: {output_dir}")
    print("=" * 50)
    print("\nNext: python 04_prepare_yolo_dataset.py")


if __name__ == '__main__':
    main()
