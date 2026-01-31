# Copilot Instructions: YOLOv11 Dev Board Identifier

## Project Overview
Real-time development board identification system using YOLOv11. Point a camera at any dev board to get instant identification with specs (processor, memory, features, price). Supports 15+ boards from Seeed, Arduino, Raspberry Pi, Espressif, and NVIDIA. Optimized for NVIDIA Jetson deployment with TensorRT export.

## Architecture & Data Flow

**Sequential Pipeline (scripts/01-06):**
```
01_scrape_images → 02_manual_capture → 03_augment_dataset → 04_prepare_yolo_dataset → 05_train_model → 06_run_inference
     ↓                    ↓                    ↓                       ↓                      ↓                ↓
  Web images       Webcam captures      100+ images/class      YOLO format (train/val/test)  best.pt     Real-time detect
```

**Key Classes:**
- `ImageScraper` (01): Web scraping with deduplication, rate limiting, attribution logging to `dataset/raw/attribution.json`
- `ManualCapture` (02): Webcam interface with burst/timed capture modes
- `DataAugmenter` (03): Parallel augmentation using `ProcessPoolExecutor` (4 workers default) - rotation, scale, brightness, flip, noise
- `DevBoardInference` (06): Real-time YOLO inference with specs overlay and MQTT publishing
- `BoardSpecs` (06): Loads metadata from `config/boards.yaml`

**Config-Driven System:** All boards, specs, training params, augmentation settings in `config/boards.yaml` - this is the single source of truth.

## Critical Implementation Patterns

**YOLO Annotation Format:**
- Format: `<class_id> <x_center> <y_center> <width> <height>` (all normalized 0-1)
- Script 04 auto-generates bounding boxes assuming centered subject with 0.1 margin
- **Production caveat:** Auto-generated boxes are approximations. For accurate annotations, use Roboflow or manual labeling

**Detection Stabilization (06_run_inference.py):**
- Requires 5 consecutive frames of same detection before announcing (`self.stable_frames`)
- Prevents flickering between similar boards
- Pattern: Track last detection + frame counter, only update overlay when stable

**Parallel Processing:**
- Augmentation uses `ProcessPoolExecutor` for parallel image processing
- Always use 4 workers: `ProcessPoolExecutor(max_workers=4)`
- Each worker processes one class directory independently

**Dataset Structure Convention:**
- Raw images: `dataset/raw/{class_name}/` (preserve originals)
- Augmented: `dataset/augmented/{class_name}/` (100+ per class target)
- YOLO format: `dataset/yolo/images/{train,val,test}/` + `labels/` directories
- Train/val/test split: 70/20/10 ratio in script 04

## Common Workflows

**Running Pipeline:**
```bash
cd scripts && ./run_pipeline.sh                    # Full pipeline
./run_pipeline.sh --demo                           # Demo only (requires trained model)
./run_pipeline.sh --train-only                     # Skip data collection
./run_pipeline.sh --model yolo11m --epochs 150     # Custom training params
```

**Individual Steps (for debugging):**
```bash
python3 scripts/03_augment_dataset.py --target-per-class 150
python3 scripts/05_train_model.py --model yolo11s --epochs 100
python3 scripts/06_run_inference.py --model runs/devboard_*/weights/best.pt --mqtt localhost:1883
```

**Model Training:**
- Default: yolo11n (nano), 100 epochs, batch=16, img_size=640
- FP16 training enabled by default for Jetson optimization: `model.train(..., half=True)`
- Outputs to `runs/devboard_{model}_{timestamp}/weights/best.pt`
- Resume training: `--resume` flag looks for `runs/last.pt`

**Adding New Board Class:**
1. Add entry to `config/boards.yaml` with display_name, manufacturer, specs, image_sources
2. Create directory `dataset/raw/{new_class_name}/`
3. Add images (run script 01 or 02)
4. Rerun pipeline from script 03 onwards

## Deployment & Integration

**MQTT Integration:**
- Topic pattern: `devboard/detection/{class_name}`
- Payload: JSON with class_name, confidence, timestamp, specs
- Usage: `--mqtt localhost:1883` flag in script 06

**TensorRT Export (Jetson optimization):**
```python
model.export(format='engine', half=True, device=0)  # FP16 precision
```

**Inference Modes:**
- PyTorch: `--model best.pt` (slower, CPU/GPU)
- TensorRT: `--model best.engine` (fastest, GPU only)
- ONNX: `--model best.onnx` (portable)

## Code Conventions

- All scripts use `argparse` for CLI args - check `--help` for options
- Logging via stdlib `logging` module (INFO level default)
- Path handling: `pathlib.Path` for cross-platform compatibility
- Error handling: Check for optional dependencies (cv2, ultralytics, paho-mqtt) with try/except imports and set `*_AVAILABLE` flags
- Scripts 01-06 are numbered for sequential execution but can run independently
- Always use absolute paths when constructing file references across directories

## Debugging Tips

- Check `dataset/raw/attribution.json` for image source tracking
- View training metrics: `runs/devboard_*/results.csv`
- Validate annotations: Open `.txt` files in `dataset/yolo/labels/` - should match image count
- Test augmentation on single class: `python3 03_augment_dataset.py --target-per-class 10`
- Inference FPS issues: Check GPU availability with `torch.cuda.is_available()`
- MQTT not publishing: Verify `MQTT_AVAILABLE` flag and broker address
