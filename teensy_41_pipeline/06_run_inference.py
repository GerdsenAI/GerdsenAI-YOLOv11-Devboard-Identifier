#!/usr/bin/env python3
"""
06_run_inference.py - Run Teensy 4.1 Detection Inference

Run the trained model on images, video, or webcam.
Outputs annotated images/video with detection boxes.

Usage:
    python 06_run_inference.py --source image.jpg
    python 06_run_inference.py --source ./test_images/
    python 06_run_inference.py --source 0  # Webcam
    python 06_run_inference.py --source video.mp4
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default to most recent trained model
RUNS_DIR = Path(__file__).parent / 'runs'
OUTPUT_DIR = Path(__file__).parent / 'output'


def find_latest_model() -> Path:
    """Find the most recently trained model."""
    if not RUNS_DIR.exists():
        return None

    runs = sorted(RUNS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
    for run in runs:
        best = run / 'weights' / 'best.pt'
        if best.exists():
            return best

    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run Teensy 4.1 inference')
    parser.add_argument('--model', type=str, default=None, help='Model weights path')
    parser.add_argument('--source', type=str, required=True,
                        help='Image, directory, video, or webcam (0)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--save', action='store_true', default=True, help='Save results')
    parser.add_argument('--show', action='store_true', help='Display results')
    parser.add_argument('--device', type=str, default='', help='Device (cuda, mps, cpu)')
    args = parser.parse_args()

    # Find model
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = find_latest_model()

    if not model_path or not model_path.exists():
        logger.error("No trained model found.")
        logger.error("Either specify --model or run 05_train_model.py first")
        sys.exit(1)

    # Import ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger.info("=" * 50)
    logger.info("TEENSY 4.1 INFERENCE")
    logger.info("=" * 50)
    logger.info(f"Model: {model_path}")
    logger.info(f"Source: {args.source}")
    logger.info(f"Confidence: {args.conf}")
    logger.info("=" * 50)

    # Load model
    model = YOLO(str(model_path))

    # Run inference
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=args.save,
        show=args.show,
        device=args.device if args.device else None,
        project=str(OUTPUT_DIR),
        name=f"detect_{timestamp}",
        exist_ok=True,
    )

    # Summary
    detections = 0
    for r in results:
        detections += len(r.boxes)

    output_path = OUTPUT_DIR / f"detect_{timestamp}"

    print("\n" + "=" * 50)
    print("INFERENCE COMPLETE")
    print("=" * 50)
    print(f"Images processed: {len(results)}")
    print(f"Detections: {detections}")
    if args.save:
        print(f"Results saved to: {output_path}")
    print("=" * 50)

    # Print detection details
    if detections > 0:
        print("\nDetections:")
        for i, r in enumerate(results):
            if len(r.boxes) > 0:
                source_name = Path(r.path).name if hasattr(r, 'path') else f"image_{i}"
                print(f"  {source_name}:")
                for box in r.boxes:
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    print(f"    - teensy_41: {conf:.2%} confidence")


if __name__ == '__main__':
    main()
