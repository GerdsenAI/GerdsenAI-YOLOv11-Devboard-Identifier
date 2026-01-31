# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLOv11-based development board identification system. Point a camera at a dev board to get real-time identification with specs overlay (processor, memory, features, price). Supports 15+ boards from Seeed, Arduino, Raspberry Pi, Espressif, and NVIDIA. Optimized for NVIDIA Jetson deployment with TensorRT export.

## Commands

### Setup
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Full Pipeline
```bash
cd scripts && ./run_pipeline.sh                    # Complete pipeline
./run_pipeline.sh --demo                           # Demo only (requires trained model)
./run_pipeline.sh --train-only                     # Skip data collection
./run_pipeline.sh --model yolo11m --epochs 150    # Custom training params
```

### Individual Steps
```bash
python3 scripts/01_scrape_images.py --config config/boards.yaml
python3 scripts/02_manual_capture.py --class arduino_uno
python3 scripts/03_augment_dataset.py --target-per-class 150
python3 scripts/04_prepare_yolo_dataset.py
python3 scripts/05_train_model.py --model yolo11s --epochs 100
python3 scripts/06_run_inference.py --model models/devboard_best.pt
```

### With MQTT
```bash
python3 scripts/06_run_inference.py --model models/devboard_best.pt --mqtt localhost:1883
```

## Architecture

**Pipeline Flow:**
```
01_scrape_images.py -> 02_manual_capture.py -> 03_augment_dataset.py
       |                      |                       |
   Web images          Webcam captures         100+ images/class
                                                      |
                                          04_prepare_yolo_dataset.py
                                                      |
                                          YOLO format (train/val/test)
                                                      |
                                          05_train_model.py
                                                      |
                                          runs/devboard_*/weights/best.pt
                                                      |
                                          06_run_inference.py
                                                      |
                                          Real-time detection + MQTT
```

**Key Classes:**
- `ImageScraper` (01): Web scraping with deduplication, rate limiting, attribution logging
- `ManualCapture` (02): Webcam interface with burst/timed capture modes
- `DataAugmenter` (03): Parallel augmentation (rotation, scale, brightness, flip, noise)
- `DevBoardInference` (06): Real-time YOLO inference with specs overlay and MQTT publishing
- `BoardSpecs` (06): Loads board metadata from config/boards.yaml

## Configuration

`config/boards.yaml` defines:
- Board classes with display names, manufacturers, specs (processor, memory, features, price)
- Image source URLs for web scraping
- Training parameters: model_base (yolo11n), image_size (640), epochs (100)
- Augmentation settings and deployment config (confidence threshold, NMS, TensorRT precision)

## Key Implementation Notes

- Auto-generated bounding boxes in `04_prepare_yolo_dataset.py` are approximate (assumes centered subject). For production, use Roboflow for manual annotation.
- Detection stabilization: 5 consecutive frames before announcing (prevents flickering)
- FP16 training enabled by default for Jetson optimization
- YOLO annotation format: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0-1)
- Augmentation uses ProcessPoolExecutor (4 workers default)
