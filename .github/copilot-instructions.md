# GerdsenAI YOLOv11 Devboard Identifier - AI Agent Instructions

## Project Overview
YOLOv11-based real-time computer vision system for identifying 17 development board classes (Seeed, Arduino, Raspberry Pi, ESP32, Jetson, Teensy). Designed for NVIDIA Jetson deployment with MQTT integration for Site Appliance connectivity.

## Architecture & Data Flow

### Pipeline Stages (Sequential)
1. **Data Collection** → `scripts/01_scrape_images.py` scrapes from manufacturer sites (PJRC, Adafruit, SparkFun) into `dataset/raw/<board_name>/`
2. **Manual Capture** → `scripts/02_manual_capture.py` (optional) captures webcam images to supplement dataset
3. **Augmentation** → `scripts/03_augment_dataset.py` expands from ~16 to 100+ images per class via rotation, brightness, contrast, flipping (constrained ranges to prevent over-darkness)
4. **YOLO Prep** → `scripts/04_prepare_yolo_dataset.py` creates train/val/test splits (70/20/10) and auto-generates bounding box annotations
5. **Training** → `scripts/05_train_model.py` trains YOLOv11 on dataset, saves to `runs/devboard_yolo11n_<timestamp>/`
6. **Inference** → `scripts/06_run_inference.py` runs real-time detection with specs overlay from `config/boards.yaml`

### Critical Annotation Behavior
**Auto-generated bounding boxes** are placeholders (centered, 80% coverage). For production, proper annotation tools (Roboflow) are needed. YOLO format: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0-1).

### Configuration Source of Truth
`config/boards.yaml` drives everything:
- Board classes (keys become class names)
- Specifications (processor, memory, features, price)
- Image scraping sources
- Display names for inference overlay

Class order in YAML determines class IDs in YOLO dataset. **Never reorder without regenerating dataset.**

## Developer Workflows

### Full Pipeline Execution
```bash
cd scripts
./run_pipeline.sh                    # Full pipeline
./run_pipeline.sh --skip-scrape      # Skip web scraping
./run_pipeline.sh --train-only       # Only training
./run_pipeline.sh --demo             # Demo with existing model
```

### Python Environment Setup
**Always use virtual environment** - project designed for isolated deps:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training Commands
```bash
python 05_train_model.py --data ../dataset/yolo/data.yaml --model yolo11n --epochs 100
python 05_train_model.py --resume  # Resume from last checkpoint
```

### Inference Commands
```bash
python 06_run_inference.py --model ../runs/best.pt --camera 0
python 06_run_inference.py --model ../runs/best.engine  # TensorRT for Jetson
python 06_run_inference.py --mqtt localhost:1883  # With MQTT publishing
```

### TensorRT Export (Jetson Optimization)
```bash
python 05_train_model.py --export engine --model-path models/devboard_best.pt
```

## Project-Specific Conventions

### File Naming Patterns
- Augmented images: `<board_name>_<index>.jpg` (e.g., `arduino_uno_0042.jpg`)
- YOLO labels match image names with `.txt` extension
- Training runs: `devboard_<model>_<YYYYMMDD_HHMMSS>/`

### Class Name Format
Use **underscores** with lowercase: `seeed_xiao_esp32s3`, `raspberry_pi_zero_2w`, `jetson_orin_nano`. This matches YAML keys and directory names throughout the pipeline.

### Dataset Structure
```
dataset/
├── raw/              # Original scraped images by class
├── augmented/        # Expanded dataset (100+ per class)
└── yolo/            # YOLO-formatted with train/val/test splits
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/      # Same structure as images/
```

### Augmentation Constraints
Script 03 uses **constrained ranges** to prevent issues:
- Brightness: 0.85-1.15 (avoids too dark/bright)
- Contrast: 0.9-1.1
- Rotation: ±12° (prevents excessive distortion)
- Edge color fill for rotations (not black borders)

## Integration Points

### MQTT Publishing (Site Appliance)
Inference script publishes to `devboard/detection` topic:
```json
{
  "board": "seeed_xiao_esp32s3",
  "confidence": 0.94,
  "timestamp": "2026-01-30T15:42:00Z",
  "specs": { "display_name": "...", "processor": "...", "memory": "..." }
}
```
Enable with `--mqtt <host>:<port>` flag. Requires `paho-mqtt` package.

### Board Specs Overlay
`BoardSpecs` class in `06_run_inference.py` loads from `config/boards.yaml` and formats for overlay display. Update YAML to change displayed information.

## Common Pitfalls & Solutions

### Script Execution Directory
All scripts assume execution from `scripts/` directory with relative paths to parent. If running from project root, adjust paths or use `run_pipeline.sh`.

### Missing Dependencies
Scripts gracefully degrade:
- No `ultralytics`: Training/inference unavailable
- No `cv2`: Inference unavailable
- No `paho-mqtt`: MQTT disabled but inference continues

### CUDA/GPU Detection
Training auto-detects GPU via `torch.cuda.is_available()`. On Jetson, ensure CUDA toolkit installed.

### Dataset Class Mismatch
If adding/removing boards from `config/boards.yaml`, **regenerate entire YOLO dataset** (script 04) to ensure class IDs match.

## Key Files Reference
- [config/boards.yaml](../config/boards.yaml) - Single source of truth for all board metadata
- [scripts/run_pipeline.sh](../scripts/run_pipeline.sh) - Orchestrates full workflow with CLI options
- [dataset/yolo/data.yaml](../dataset/yolo/data.yaml) - YOLO dataset config (auto-generated)
- [scripts/05_train_model.py](../scripts/05_train_model.py) - Training logic with auto device detection
- [scripts/06_run_inference.py](../scripts/06_run_inference.py) - Real-time inference with BoardSpecs integration
