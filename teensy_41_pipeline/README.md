# Teensy 4.1 Detection Pipeline

Standalone YOLOv11 pipeline for Teensy 4.1 devboard detection.
Optimized for demos and testing.

## Quick Start

```bash
# 1. Already have 41 raw images in data/raw/

# 2. Augment to 200 images
python 03_augment_dataset.py --target 200

# 3. Prepare YOLO dataset
python 04_prepare_yolo_dataset.py

# 4. Train model (~30 min on GPU, longer on CPU)
python 05_train_model.py --epochs 30

# 5. Run inference
python 06_run_inference.py --source test_image.jpg
python 06_run_inference.py --source 0  # Webcam
```

## Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `01_scrape_images.py` | Scrape Teensy 4.1 images from PJRC, Adafruit, SparkFun |
| `02_manual_capture.py` | Process manually downloaded images (convert, rename, dedupe) |
| `03_augment_dataset.py` | Expand dataset with rotation, brightness, flip, noise |
| `04_prepare_yolo_dataset.py` | Create YOLO format with train/val/test splits |
| `05_train_model.py` | Train YOLOv11 model |
| `06_run_inference.py` | Run detection on images, video, or webcam |

## Directory Structure

```
teensy_41_pipeline/
├── data/
│   ├── raw/           # Original images (41 curated)
│   ├── augmented/     # After augmentation (200+)
│   └── yolo/          # YOLO format dataset
├── runs/              # Training runs
├── output/            # Inference results
└── *.py               # Pipeline scripts
```

## Current Dataset

- 41 curated Teensy 4.1 board images
- Sources: PJRC, Adafruit, SparkFun, manual captures
- Cleaned: removed pinout diagrams, accessories, wrong products

## Augmentation Features

- Rotation (edge-color fill, not black)
- Brightness/contrast adjustment (constrained to prevent dark images)
- Horizontal flip
- Scale variations
- Gaussian noise and blur
- Automatic brightness validation (rejects avg < 30 or > 240)

## Training Tips

- GPU recommended (CUDA or Apple MPS)
- CPU training: reduce epochs to 10-15
- For quick demo: `--epochs 15 --batch 4`
- For production: `--epochs 50 --batch 16`

## Requirements

```bash
pip install ultralytics pillow numpy requests beautifulsoup4
```
