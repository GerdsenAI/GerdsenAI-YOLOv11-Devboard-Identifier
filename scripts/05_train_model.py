#!/usr/bin/env python3
"""
05_train_model.py - Train YOLOv11 model for dev board identification

This script trains a YOLOv11 model on the prepared dataset.
Optimized for NVIDIA Jetson deployment.

Usage:
    python 05_train_model.py --data ../dataset/yolo/data.yaml
    python 05_train_model.py --model yolov11s --epochs 150
    python 05_train_model.py --resume  # Resume from last checkpoint

Requirements:
    pip install ultralytics
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics not installed. Run: pip install ultralytics")


def get_device():
    """Detect best available device."""
    import torch
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {device_name}")
        return 0
    else:
        logger.info("Using CPU (training will be slow)")
        return 'cpu'


def train_model(data_yaml: str, model_name: str = 'yolov11n',
                epochs: int = 100, batch_size: int = 16,
                img_size: int = 640, project_dir: str = None,
                resume: bool = False):
    """Train YOLOv11 model."""

    if not ULTRALYTICS_AVAILABLE:
        logger.error("Cannot train without ultralytics. Install with: pip install ultralytics")
        return None

    # Determine project directory
    if project_dir is None:
        project_dir = Path(__file__).parent.parent / 'runs'

    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    # Generate run name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"devboard_{model_name}_{timestamp}"

    logger.info("=" * 50)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"Model:      {model_name}")
    logger.info(f"Data:       {data_yaml}")
    logger.info(f"Epochs:     {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Image size: {img_size}")
    logger.info(f"Project:    {project_dir}")
    logger.info(f"Run name:   {run_name}")
    logger.info("=" * 50)

    # Load model
    if resume:
        # Find last checkpoint
        last_pt = project_dir / 'last.pt'
        if last_pt.exists():
            model = YOLO(str(last_pt))
            logger.info(f"Resuming from: {last_pt}")
        else:
            logger.error("No checkpoint found to resume")
            return None
    else:
        # Load pretrained model
        model = YOLO(f'{model_name}.pt')
        logger.info(f"Loaded pretrained: {model_name}.pt")

    # Get device
    device = get_device()

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=str(project_dir),
        name=run_name,

        # Optimization
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # Augmentation (in addition to pre-augmented data)
        augment=True,
        mosaic=0.5,
        mixup=0.1,
        copy_paste=0.0,

        # Training settings
        patience=20,          # Early stopping patience
        save=True,
        save_period=10,       # Save checkpoint every N epochs
        cache=True,           # Cache images for faster training
        workers=4,

        # Jetson optimization
        half=True,            # FP16 training

        # Visualization
        plots=True,
        verbose=True,
    )

    # Get best model path
    best_model = project_dir / run_name / 'weights' / 'best.pt'

    logger.info("\n" + "=" * 50)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Best model: {best_model}")
    logger.info(f"Results:    {project_dir / run_name}")
    logger.info("=" * 50)

    return str(best_model)


def validate_model(model_path: str, data_yaml: str):
    """Validate trained model."""
    if not ULTRALYTICS_AVAILABLE:
        return None

    logger.info(f"Validating: {model_path}")

    model = YOLO(model_path)
    results = model.val(data=data_yaml)

    logger.info("\nValidation Results:")
    logger.info(f"  mAP50:    {results.box.map50:.4f}")
    logger.info(f"  mAP50-95: {results.box.map:.4f}")

    return results


def export_model(model_path: str, format: str = 'engine'):
    """Export model for deployment."""
    if not ULTRALYTICS_AVAILABLE:
        return None

    logger.info(f"Exporting {model_path} to {format} format...")

    model = YOLO(model_path)

    if format == 'engine':
        # TensorRT for Jetson
        exported = model.export(
            format='engine',
            device=0,
            half=True,
            imgsz=640,
            simplify=True,
        )
    elif format == 'onnx':
        exported = model.export(
            format='onnx',
            simplify=True,
            dynamic=False,
            imgsz=640,
        )
    else:
        exported = model.export(format=format)

    logger.info(f"Exported: {exported}")
    return exported


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv11 for dev board identification')
    parser.add_argument('--data', type=str, default='../dataset/yolo/data.yaml',
                        help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov11n',
                        choices=['yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x'],
                        help='Base model to use')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--project', type=str, default='../runs',
                        help='Project directory for outputs')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate existing model')
    parser.add_argument('--export', type=str, choices=['engine', 'onnx', 'torchscript'],
                        help='Export model after training')
    parser.add_argument('--model-path', type=str,
                        help='Path to model for validation/export')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_yaml = str((script_dir / args.data).absolute())
    project_dir = str((script_dir / args.project).absolute())

    if not Path(data_yaml).exists():
        logger.error(f"Data config not found: {data_yaml}")
        sys.exit(1)

    if args.validate_only:
        if not args.model_path:
            logger.error("--model-path required for validation")
            sys.exit(1)
        validate_model(args.model_path, data_yaml)
        return

    if args.export and args.model_path:
        export_model(args.model_path, args.export)
        return

    # Train
    best_model = train_model(
        data_yaml=data_yaml,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        project_dir=project_dir,
        resume=args.resume,
    )

    if best_model:
        # Validate
        validate_model(best_model, data_yaml)

        # Export if requested
        if args.export:
            export_model(best_model, args.export)


if __name__ == '__main__':
    main()
