#!/usr/bin/env python3
"""
05_train_model.py - Train YOLOv11 on Teensy 4.1 Dataset

Trains a YOLOv11 model specifically for Teensy 4.1 detection.
Single class, optimized for quick training and demo purposes.

Usage:
    python 05_train_model.py
    python 05_train_model.py --epochs 50 --batch 16
    python 05_train_model.py --model yolo11s.pt  # Use small model
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_YAML = Path(__file__).parent / 'data' / 'yolo' / 'data.yaml'
OUTPUT_DIR = Path(__file__).parent / 'runs'


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train YOLOv11 on Teensy 4.1')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        help='Base model (yolo11n.pt, yolo11s.pt, yolo11m.pt)')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--data', type=str, default=None, help='Path to data.yaml')
    parser.add_argument('--device', type=str, default='', help='Device (cuda, mps, cpu)')
    args = parser.parse_args()

    data_yaml = Path(args.data) if args.data else DATA_YAML

    if not data_yaml.exists():
        logger.error(f"data.yaml not found: {data_yaml}")
        logger.error("Run 04_prepare_yolo_dataset.py first")
        sys.exit(1)

    # Try to import ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"teensy41_{timestamp}"

    logger.info("=" * 50)
    logger.info("TEENSY 4.1 YOLOV11 TRAINING")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch}")
    logger.info(f"Image size: {args.imgsz}")
    logger.info(f"Data: {data_yaml}")
    logger.info("=" * 50)

    # Load model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=run_name,
        project=str(OUTPUT_DIR),
        exist_ok=True,
        device=args.device if args.device else None,
        patience=10,  # Early stopping
        save=True,
        plots=True,
        verbose=True,
    )

    # Find best weights
    best_weights = OUTPUT_DIR / run_name / 'weights' / 'best.pt'

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Best weights: {best_weights}")
    print(f"Run directory: {OUTPUT_DIR / run_name}")
    print("=" * 50)
    print("\nNext: python 06_run_inference.py --model", best_weights)


if __name__ == '__main__':
    main()
