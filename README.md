# Dev Board Identifier

**"What Board Is This?"** - AI-powered development board identification using YOLOv11.

Point a camera at any development board and instantly get identification + specs.

## Features

- Real-time board detection via webcam
- 15+ board classes (Seeed, Arduino, Raspberry Pi, ESP32, Jetson)
- Specs overlay with processor, memory, features, price
- MQTT integration for Site Appliance
- Optimized for NVIDIA Jetson deployment

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
cd scripts
./run_pipeline.sh
```

This will:
1. Scrape board images from manufacturer sites
2. (Optional) Launch manual capture interface
3. Augment dataset to 100+ images per class
4. Prepare YOLO-format dataset
5. Train YOLOv11 model
6. Launch live demo

### 3. Run Demo Only

If you already have a trained model:

```bash
./run_pipeline.sh --demo
```

Or directly:

```bash
python scripts/06_run_inference.py --model models/devboard_best.pt
```

## Supported Boards

### Seeed Studio
- XIAO ESP32-S3
- XIAO RP2040
- XIAO nRF52840
- Wio Terminal

### Arduino
- Uno R3/R4
- Nano
- Mega 2560

### Raspberry Pi
- Pi 4 Model B
- Pi 5
- Pico / Pico W
- Zero 2 W

### Espressif
- ESP32-DevKitC
- ESP32-CAM
- ESP32-S3-DevKitC

### NVIDIA
- Jetson Orin Nano
- Jetson Orin NX

## Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `01_scrape_images.py` | Collect images from web |
| `02_manual_capture.py` | Webcam capture interface |
| `03_augment_dataset.py` | Data augmentation |
| `04_prepare_yolo_dataset.py` | Convert to YOLO format |
| `05_train_model.py` | Train YOLOv11 |
| `06_run_inference.py` | Real-time detection |
| `run_pipeline.sh` | Master automation script |

## Configuration

Edit `config/boards.yaml` to:
- Add/remove board classes
- Update specifications
- Adjust training parameters
- Configure augmentation

## Site Appliance Integration

The detector publishes to MQTT topic `devboard/detection`:

```json
{
  "board": "seeed_xiao_esp32s3",
  "confidence": 0.94,
  "timestamp": "2026-01-30T15:42:00Z",
  "specs": {
    "display_name": "Seeed XIAO ESP32-S3",
    "processor": "ESP32-S3 @ 240MHz",
    "memory": "8MB Flash, 512KB SRAM"
  }
}
```

## Jetson Deployment

Export to TensorRT for faster inference:

```bash
python scripts/05_train_model.py --export engine --model-path models/devboard_best.pt
```

Then run with `.engine` file:

```bash
python scripts/06_run_inference.py --model models/devboard_best.engine
```

## Tips for Better Accuracy

1. **More data is better** - Use manual capture to add 20+ images per board
2. **Vary backgrounds** - Capture on different surfaces (desk, white paper, etc.)
3. **Vary angles** - Front, slight rotation, different distances
4. **Use Roboflow** - For proper bounding box annotation
5. **Increase model size** - Use `yolov11s` or `yolov11m` if accuracy is low

## Demo Recording

For Seeed certification demo:

```bash
python scripts/06_run_inference.py \
  --model models/devboard_best.pt \
  --record demo_video.mp4
```

Or press `R` during live inference to toggle recording.

## License

MIT License - See LICENSE file.

## Credits

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Seeed Studio](https://www.seeedstudio.com/)
- [GerdsenAI](https://github.com/GerdsenAI)
