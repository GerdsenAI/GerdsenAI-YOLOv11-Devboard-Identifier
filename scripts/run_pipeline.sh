#!/bin/bash
# ============================================================
# Dev Board Identifier - Automated Pipeline
# ============================================================
# This script runs the complete pipeline from data collection
# to model deployment.
#
# Usage:
#   ./run_pipeline.sh              # Full pipeline
#   ./run_pipeline.sh --skip-scrape # Skip web scraping
#   ./run_pipeline.sh --train-only  # Only training
#   ./run_pipeline.sh --demo        # Run demo only
#
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="$PROJECT_DIR/config/boards.yaml"
RAW_DATASET="$PROJECT_DIR/dataset/raw"
AUG_DATASET="$PROJECT_DIR/dataset/augmented"
YOLO_DATASET="$PROJECT_DIR/dataset/yolo"
RUNS_DIR="$PROJECT_DIR/runs"
MODELS_DIR="$PROJECT_DIR/models"

# Defaults
SKIP_SCRAPE=false
TRAIN_ONLY=false
DEMO_ONLY=false
TARGET_IMAGES=100
MODEL_TYPE="yolov11n"
EPOCHS=100

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-scrape)
            SKIP_SCRAPE=true
            shift
            ;;
        --train-only)
            TRAIN_ONLY=true
            shift
            ;;
        --demo)
            DEMO_ONLY=true
            shift
            ;;
        --model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --target-images)
            TARGET_IMAGES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-scrape     Skip web scraping step"
            echo "  --train-only      Only run training (assume data exists)"
            echo "  --demo            Run demo only (assume model exists)"
            echo "  --model TYPE      Model type (yolov11n/s/m/l, default: yolov11n)"
            echo "  --epochs N        Training epochs (default: 100)"
            echo "  --target-images N Images per class after augmentation (default: 100)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Functions
print_header() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}\n"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    print_header "Checking Dependencies"

    # Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        exit 1
    fi
    print_step "Python 3 found: $(python3 --version)"

    # Required packages
    REQUIRED_PACKAGES="yaml PIL cv2 numpy"
    for pkg in $REQUIRED_PACKAGES; do
        if python3 -c "import $pkg" 2>/dev/null; then
            print_step "$pkg available"
        else
            print_warning "$pkg not found - installing..."
        fi
    done

    # Check ultralytics
    if python3 -c "from ultralytics import YOLO" 2>/dev/null; then
        print_step "Ultralytics YOLO available"
    else
        print_warning "Installing ultralytics..."
        pip install ultralytics -q
    fi
}

run_scraping() {
    print_header "Step 1: Web Scraping"

    if [ "$SKIP_SCRAPE" = true ]; then
        print_warning "Skipping web scraping"
        return
    fi

    print_step "Scraping board images from manufacturer sites..."
    python3 "$SCRIPT_DIR/01_scrape_images.py" \
        --config "$CONFIG_FILE" \
        --output "$RAW_DATASET"

    # Count images
    if [ -d "$RAW_DATASET" ]; then
        COUNT=$(find "$RAW_DATASET" -name "*.jpg" -o -name "*.png" | wc -l)
        print_step "Downloaded $COUNT images"
    fi
}

run_manual_capture() {
    print_header "Step 2: Manual Capture (Optional)"

    echo -e "${YELLOW}Would you like to capture additional images manually? (y/N)${NC}"
    read -r response

    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_step "Starting manual capture interface..."
        python3 "$SCRIPT_DIR/02_manual_capture.py" \
            --config "$CONFIG_FILE" \
            --output "$RAW_DATASET"
    else
        print_step "Skipping manual capture"
    fi
}

run_augmentation() {
    print_header "Step 3: Data Augmentation"

    print_step "Augmenting dataset to $TARGET_IMAGES images per class..."
    python3 "$SCRIPT_DIR/03_augment_dataset.py" \
        --input "$RAW_DATASET" \
        --output "$AUG_DATASET" \
        --config "$CONFIG_FILE" \
        --target-per-class "$TARGET_IMAGES"
}

run_yolo_prep() {
    print_header "Step 4: Prepare YOLO Dataset"

    print_step "Converting to YOLO format..."
    python3 "$SCRIPT_DIR/04_prepare_yolo_dataset.py" \
        --input "$AUG_DATASET" \
        --output "$YOLO_DATASET" \
        --config "$CONFIG_FILE"

    print_warning "NOTE: For production, annotate images properly in Roboflow"
}

run_training() {
    print_header "Step 5: Model Training"

    print_step "Training $MODEL_TYPE for $EPOCHS epochs..."
    python3 "$SCRIPT_DIR/05_train_model.py" \
        --data "$YOLO_DATASET/data.yaml" \
        --model "$MODEL_TYPE" \
        --epochs "$EPOCHS" \
        --project "$RUNS_DIR"

    # Find best model
    BEST_MODEL=$(find "$RUNS_DIR" -name "best.pt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")

    if [ -n "$BEST_MODEL" ]; then
        print_step "Best model: $BEST_MODEL"

        # Copy to models directory
        mkdir -p "$MODELS_DIR"
        cp "$BEST_MODEL" "$MODELS_DIR/devboard_best.pt"
        print_step "Copied to: $MODELS_DIR/devboard_best.pt"
    fi
}

run_demo() {
    print_header "Step 6: Run Demo"

    # Find model
    MODEL_PATH="$MODELS_DIR/devboard_best.pt"
    if [ ! -f "$MODEL_PATH" ]; then
        MODEL_PATH=$(find "$RUNS_DIR" -name "best.pt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")
    fi

    if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
        print_error "No trained model found!"
        exit 1
    fi

    print_step "Running inference with: $MODEL_PATH"
    python3 "$SCRIPT_DIR/06_run_inference.py" \
        --model "$MODEL_PATH" \
        --config "$CONFIG_FILE"
}

# Main execution
print_header "Dev Board Identifier Pipeline"
echo "Project: $PROJECT_DIR"
echo "Config:  $CONFIG_FILE"
echo ""

if [ "$DEMO_ONLY" = true ]; then
    run_demo
    exit 0
fi

if [ "$TRAIN_ONLY" = true ]; then
    check_dependencies
    run_training
    run_demo
    exit 0
fi

# Full pipeline
check_dependencies
run_scraping
run_manual_capture
run_augmentation
run_yolo_prep
run_training
run_demo

print_header "Pipeline Complete!"
echo -e "${GREEN}Your dev board identifier is ready!${NC}"
echo ""
echo "Next steps:"
echo "  1. For better accuracy, annotate images in Roboflow"
echo "  2. Export to TensorRT for faster Jetson inference"
echo "  3. Integrate with Site Appliance via MQTT"
echo ""
