#!/usr/bin/env python3
"""
02_manual_capture.py - Capture images from webcam for dataset

This script provides a simple interface to capture board images using a webcam.
It's the most reliable way to get high-quality, consistent training data.

Features:
- Live preview with keyboard shortcuts
- Auto-naming with class labels
- Multiple capture modes (single, burst, timed)
- Optional background removal

Usage:
    python 02_manual_capture.py --class arduino_uno --output ../dataset/raw
    python 02_manual_capture.py --class seeed_xiao_esp32s3 --burst 5

Controls:
    SPACE - Capture image
    B - Burst mode (5 images)
    T - Timed capture (3 seconds)
    N - Next class
    Q - Quit
"""

import cv2
import os
import sys
import yaml
import argparse
import time
from pathlib import Path
from datetime import datetime

# Camera settings
CAMERA_INDEX = 0  # Default webcam
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720


class ManualCapture:
    def __init__(self, config_path: str, output_dir: str, initial_class: str = None):
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.board_classes = list(self.config.get('boards', {}).keys())
        self.current_class_idx = 0

        if initial_class and initial_class in self.board_classes:
            self.current_class_idx = self.board_classes.index(initial_class)

        self.capture_count = {}
        for cls in self.board_classes:
            cls_dir = self.output_dir / cls
            cls_dir.mkdir(exist_ok=True)
            # Count existing images
            self.capture_count[cls] = len(list(cls_dir.glob('*.jpg'))) + len(list(cls_dir.glob('*.png')))

        # Initialize camera
        self.cap = None

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    @property
    def current_class(self) -> str:
        return self.board_classes[self.current_class_idx]

    @property
    def current_display_name(self) -> str:
        boards = self.config.get('boards', {})
        return boards.get(self.current_class, {}).get('display_name', self.current_class)

    def start_camera(self):
        """Initialize the camera."""
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

        if not self.cap.isOpened():
            print("Error: Could not open camera")
            sys.exit(1)

        print(f"Camera initialized at {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}")

    def stop_camera(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def capture_frame(self) -> bool:
        """Capture and save a single frame."""
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame")
            return False

        # Generate filename
        cls = self.current_class
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:20]
        filename = f"{cls}_manual_{timestamp}.jpg"

        # Save to class directory
        cls_dir = self.output_dir / cls
        filepath = cls_dir / filename

        cv2.imwrite(str(filepath), frame)
        self.capture_count[cls] += 1

        print(f"Captured: {filename} (Total for {cls}: {self.capture_count[cls]})")
        return True

    def burst_capture(self, count: int = 5, delay: float = 0.3):
        """Capture multiple frames in quick succession."""
        print(f"Burst capture: {count} images...")
        for i in range(count):
            self.capture_frame()
            time.sleep(delay)
        print("Burst complete!")

    def timed_capture(self, countdown: int = 3):
        """Capture after countdown."""
        for i in range(countdown, 0, -1):
            print(f"Capturing in {i}...")
            time.sleep(1)
        self.capture_frame()

    def next_class(self):
        """Switch to next board class."""
        self.current_class_idx = (self.current_class_idx + 1) % len(self.board_classes)
        print(f"\nSwitched to: {self.current_display_name}")
        print(f"Current count: {self.capture_count[self.current_class]}")

    def prev_class(self):
        """Switch to previous board class."""
        self.current_class_idx = (self.current_class_idx - 1) % len(self.board_classes)
        print(f"\nSwitched to: {self.current_display_name}")
        print(f"Current count: {self.capture_count[self.current_class]}")

    def draw_overlay(self, frame):
        """Draw information overlay on preview frame."""
        h, w = frame.shape[:2]

        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Text info
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)

        cv2.putText(frame, f"Class: {self.current_display_name}", (20, 40), font, 0.7, color, 2)
        cv2.putText(frame, f"Count: {self.capture_count[self.current_class]}", (20, 70), font, 0.7, color, 2)
        cv2.putText(frame, f"[{self.current_class_idx + 1}/{len(self.board_classes)}]", (20, 100), font, 0.6, (200, 200, 200), 1)

        # Controls
        cv2.putText(frame, "SPACE:Capture  B:Burst  T:Timed  N:Next  P:Prev  Q:Quit", (20, 130), font, 0.5, (150, 150, 150), 1)

        # Crosshair in center
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (0, 255, 0), 1)
        cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (0, 255, 0), 1)

        return frame

    def run(self):
        """Main capture loop."""
        self.start_camera()

        print("\n" + "=" * 50)
        print("DEV BOARD IMAGE CAPTURE")
        print("=" * 50)
        print(f"Starting class: {self.current_display_name}")
        print(f"Output directory: {self.output_dir}")
        print("\nControls:")
        print("  SPACE - Capture image")
        print("  B - Burst mode (5 images)")
        print("  T - Timed capture (3 sec countdown)")
        print("  N - Next class")
        print("  P - Previous class")
        print("  Q - Quit")
        print("=" * 50 + "\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # Resize for preview
                preview = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
                preview = self.draw_overlay(preview)

                cv2.imshow('Dev Board Capture', preview)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.capture_frame()
                elif key == ord('b'):
                    self.burst_capture()
                elif key == ord('t'):
                    self.timed_capture()
                elif key == ord('n'):
                    self.next_class()
                elif key == ord('p'):
                    self.prev_class()

        finally:
            self.stop_camera()

        # Print summary
        print("\n" + "=" * 50)
        print("CAPTURE SUMMARY")
        print("=" * 50)
        for cls, count in self.capture_count.items():
            print(f"  {cls}: {count} images")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Manual capture for dev board dataset')
    parser.add_argument('--config', type=str, default='../config/boards.yaml',
                        help='Path to boards.yaml config file')
    parser.add_argument('--output', type=str, default='../dataset/raw',
                        help='Output directory for images')
    parser.add_argument('--class', dest='initial_class', type=str,
                        help='Initial board class to capture')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index')
    args = parser.parse_args()

    global CAMERA_INDEX
    CAMERA_INDEX = args.camera

    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    output_dir = script_dir / args.output

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    capture = ManualCapture(str(config_path), str(output_dir), args.initial_class)
    capture.run()


if __name__ == '__main__':
    main()
