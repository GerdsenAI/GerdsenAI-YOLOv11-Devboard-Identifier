#!/usr/bin/env python3
"""
02_manual_capture.py - Teensy 4.1 Manual Image Capture Helper

Helps organize manually captured/downloaded Teensy 4.1 images.
- Converts WebP/AVIF to JPEG
- Renames to consistent format
- Validates image quality
- Removes duplicates

Usage:
    python 02_manual_capture.py --input ~/Downloads --watch
    python 02_manual_capture.py --input ./manual_images
"""

import os
import sys
import time
import hashlib
import shutil
from pathlib import Path
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / 'data' / 'raw'
MIN_WIDTH = 200
MIN_HEIGHT = 150
MIN_SIZE_BYTES = 5000


def get_image_hash(filepath: Path) -> str:
    """Generate MD5 hash from file content."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def convert_to_jpeg(input_path: Path, output_path: Path) -> bool:
    """Convert image to JPEG format."""
    try:
        img = Image.open(input_path)
        if img.mode in ('RGBA', 'P', 'LA'):
            # Convert transparency to white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        img.save(output_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        logger.error(f"Failed to convert {input_path}: {e}")
        return False


def is_valid_image(filepath: Path) -> bool:
    """Check if image meets quality requirements."""
    try:
        img = Image.open(filepath)
        w, h = img.size
        size = filepath.stat().st_size
        return w >= MIN_WIDTH and h >= MIN_HEIGHT and size >= MIN_SIZE_BYTES
    except:
        return False


def process_images(input_dir: Path, output_dir: Path) -> dict:
    """Process all images in input directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {'processed': 0, 'skipped': 0, 'converted': 0, 'duplicates': 0}
    existing_hashes = set()

    # Get hashes of existing images
    for f in output_dir.glob('*.jpg'):
        existing_hashes.add(get_image_hash(f))

    # Supported formats
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.avif', '.gif', '.bmp'}

    for filepath in input_dir.iterdir():
        if filepath.suffix.lower() not in extensions:
            continue

        # Check if valid
        if not is_valid_image(filepath):
            logger.warning(f"Skipped (too small): {filepath.name}")
            stats['skipped'] += 1
            continue

        # Convert if needed
        if filepath.suffix.lower() in {'.webp', '.avif', '.png', '.gif', '.bmp'}:
            temp_path = output_dir / f"temp_{filepath.stem}.jpg"
            if not convert_to_jpeg(filepath, temp_path):
                stats['skipped'] += 1
                continue
            source_path = temp_path
            stats['converted'] += 1
        else:
            source_path = filepath

        # Check for duplicates
        img_hash = get_image_hash(source_path)
        if img_hash in existing_hashes:
            logger.info(f"Skipped (duplicate): {filepath.name}")
            if source_path != filepath:
                source_path.unlink()
            stats['duplicates'] += 1
            continue

        # Save with consistent naming
        output_path = output_dir / f"teensy_41_{img_hash}.jpg"
        if source_path != filepath:
            shutil.move(source_path, output_path)
        else:
            shutil.copy2(filepath, output_path)

        existing_hashes.add(img_hash)
        stats['processed'] += 1
        logger.info(f"Added: {output_path.name}")

    return stats


def watch_directory(input_dir: Path, output_dir: Path, interval: int = 5):
    """Watch directory for new images."""
    logger.info(f"Watching {input_dir} for new images...")
    logger.info("Press Ctrl+C to stop")

    processed_files = set()

    try:
        while True:
            extensions = {'.jpg', '.jpeg', '.png', '.webp', '.avif'}
            current_files = {f for f in input_dir.iterdir() if f.suffix.lower() in extensions}
            new_files = current_files - processed_files

            if new_files:
                for f in new_files:
                    # Wait for file to finish downloading
                    time.sleep(1)
                    stats = process_images(input_dir, output_dir)
                    processed_files.update(new_files)

            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("\nStopped watching")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process Teensy 4.1 images')
    parser.add_argument('--input', type=str, required=True, help='Input directory')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--watch', action='store_true', help='Watch for new files')
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output) if args.output else OUTPUT_DIR

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    if args.watch:
        watch_directory(input_dir, output_dir)
    else:
        stats = process_images(input_dir, output_dir)

        print("\n" + "=" * 50)
        print("PROCESSING COMPLETE")
        print("=" * 50)
        print(f"Processed: {stats['processed']}")
        print(f"Converted: {stats['converted']}")
        print(f"Duplicates: {stats['duplicates']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"Output: {output_dir}")
        print("=" * 50)
        print("\nNext: python 03_augment_dataset.py")


if __name__ == '__main__':
    main()
