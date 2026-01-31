#!/usr/bin/env python3
"""
06_run_inference.py - Run real-time inference for dev board identification

This script runs the trained model on a live camera feed and displays
detections with board information from the LLM.

Features:
- Real-time camera inference
- Board specs overlay (from config or LLM)
- MQTT publishing for Site Appliance integration
- Video recording option

Usage:
    python 06_run_inference.py --model ../runs/best.pt
    python 06_run_inference.py --model ../runs/best.engine --camera 0
    python 06_run_inference.py --mqtt localhost:1883
"""

import os
import sys
import yaml
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False


class BoardSpecs:
    """Load and provide board specifications."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.boards = self.config.get('boards', {})

    def get_specs(self, class_name: str) -> dict:
        """Get specifications for a board class."""
        return self.boards.get(class_name, {})

    def format_specs(self, class_name: str) -> list:
        """Format specs as display lines."""
        specs = self.get_specs(class_name)
        if not specs:
            return [class_name]

        lines = [
            specs.get('display_name', class_name),
            specs.get('processor', ''),
            specs.get('memory', ''),
        ]

        features = specs.get('features', [])
        if features:
            lines.append(features[0] if features else '')

        price = specs.get('price_usd')
        if price:
            lines.append(f"~${price:.2f} USD")

        return [l for l in lines if l]


class DevBoardInference:
    """Real-time dev board detection and display."""

    def __init__(self, model_path: str, config_path: str,
                 camera_index: int = 0, confidence: float = 0.7,
                 mqtt_broker: str = None):

        self.model = YOLO(model_path)
        self.specs = BoardSpecs(config_path)
        self.camera_index = camera_index
        self.confidence = confidence

        # MQTT setup
        self.mqtt_client = None
        if mqtt_broker and MQTT_AVAILABLE:
            self._setup_mqtt(mqtt_broker)

        # Detection state
        self.last_detection = None
        self.detection_stable_count = 0
        self.stable_threshold = 5  # Frames before announcing detection

        # Video recording
        self.video_writer = None

    def _setup_mqtt(self, broker: str):
        """Setup MQTT client."""
        try:
            host, port = broker.split(':') if ':' in broker else (broker, 1883)
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.connect(host, int(port), 60)
            self.mqtt_client.loop_start()
            logger.info(f"Connected to MQTT broker: {broker}")
        except Exception as e:
            logger.warning(f"MQTT connection failed: {e}")

    def publish_detection(self, class_name: str, confidence: float):
        """Publish detection to MQTT."""
        if not self.mqtt_client:
            return

        payload = {
            'board': class_name,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'specs': self.specs.get_specs(class_name)
        }

        try:
            self.mqtt_client.publish(
                'devboard/detection',
                json.dumps(payload)
            )
        except Exception as e:
            logger.debug(f"MQTT publish failed: {e}")

    def draw_detection(self, frame, box, class_name: str, confidence: float):
        """Draw detection box and specs overlay."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)

        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Get specs
        spec_lines = self.specs.format_specs(class_name)

        # Draw specs panel
        panel_width = 350
        panel_height = 30 + len(spec_lines) * 25
        panel_x = w - panel_width - 20
        panel_y = 20

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Draw border
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      color, 2)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = panel_y + 25

        # Title (bold effect with multiple draws)
        cv2.putText(frame, spec_lines[0], (panel_x + 10, y_offset),
                    font, 0.6, (255, 255, 255), 2)
        y_offset += 25

        # Specs
        for line in spec_lines[1:]:
            cv2.putText(frame, line, (panel_x + 10, y_offset),
                        font, 0.45, (200, 200, 200), 1)
            y_offset += 22

        # Confidence bar
        conf_bar_width = int((panel_width - 20) * confidence)
        cv2.rectangle(frame, (panel_x + 10, y_offset),
                      (panel_x + 10 + conf_bar_width, y_offset + 8),
                      color, -1)
        cv2.putText(frame, f"{confidence:.0%}", (panel_x + panel_width - 50, y_offset + 8),
                    font, 0.4, (150, 150, 150), 1)

        # Label on box
        label = f"{class_name} {confidence:.0%}"
        (label_w, label_h), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), font, 0.5, (0, 0, 0), 1)

        return frame

    def draw_status(self, frame, fps: float, detecting: bool):
        """Draw status bar."""
        h, w = frame.shape[:2]

        # Status bar at bottom
        cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        status_color = (0, 255, 0) if detecting else (100, 100, 100)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 8), font, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "DETECTING" if detecting else "SCANNING...",
                    (w - 120, h - 8), font, 0.5, status_color, 1)

        # Recording indicator
        if self.video_writer:
            cv2.circle(frame, (w // 2, h - 15), 8, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w // 2 + 15, h - 8), font, 0.5, (0, 0, 255), 1)

        return frame

    def start_recording(self, output_path: str, fps: float, frame_size: tuple):
        """Start video recording."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        logger.info(f"Recording started: {output_path}")

    def stop_recording(self):
        """Stop video recording."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            logger.info("Recording stopped")

    def run(self, display: bool = True, record_path: str = None):
        """Main inference loop."""
        if not CV2_AVAILABLE:
            logger.error("OpenCV not available")
            return

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error(f"Could not open camera {self.camera_index}")
            return

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Camera: {frame_width}x{frame_height}")
        logger.info("Press 'q' to quit, 'r' to toggle recording")

        # Start recording if path provided
        if record_path:
            self.start_recording(record_path, 30, (frame_width, frame_height))

        fps_start = time.time()
        frame_count = 0
        fps = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Run inference
                results = self.model(frame, conf=self.confidence, verbose=False)

                # Process detections
                detecting = False
                for result in results:
                    boxes = result.boxes
                    if len(boxes) > 0:
                        # Get best detection
                        best_idx = boxes.conf.argmax()
                        box = boxes.xyxy[best_idx].cpu().numpy()
                        conf = float(boxes.conf[best_idx])
                        cls_id = int(boxes.cls[best_idx])
                        cls_name = result.names[cls_id]

                        # Track stable detection
                        if cls_name == self.last_detection:
                            self.detection_stable_count += 1
                        else:
                            self.last_detection = cls_name
                            self.detection_stable_count = 1

                        # Publish if stable
                        if self.detection_stable_count == self.stable_threshold:
                            self.publish_detection(cls_name, conf)

                        # Draw
                        frame = self.draw_detection(frame, box, cls_name, conf)
                        detecting = True

                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start)
                    fps_start = time.time()

                # Draw status
                frame = self.draw_status(frame, fps, detecting)

                # Record
                if self.video_writer:
                    self.video_writer.write(frame)

                # Display
                if display:
                    cv2.imshow('Dev Board Identifier', frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        if self.video_writer:
                            self.stop_recording()
                        else:
                            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                            self.start_recording(f'devboard_demo_{ts}.mp4', 30,
                                                 (frame_width, frame_height))

        finally:
            cap.release()
            self.stop_recording()
            cv2.destroyAllWindows()
            if self.mqtt_client:
                self.mqtt_client.loop_stop()


def main():
    parser = argparse.ArgumentParser(description='Run dev board identification')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pt or .engine)')
    parser.add_argument('--config', type=str, default='../config/boards.yaml',
                        help='Path to boards.yaml')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index')
    parser.add_argument('--confidence', type=float, default=0.7,
                        help='Detection confidence threshold')
    parser.add_argument('--mqtt', type=str,
                        help='MQTT broker address (host:port)')
    parser.add_argument('--record', type=str,
                        help='Output video file path')
    parser.add_argument('--no-display', action='store_true',
                        help='Run without display (headless)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    config_path = str((script_dir / args.config).absolute())

    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    if not Path(config_path).exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    inference = DevBoardInference(
        model_path=args.model,
        config_path=config_path,
        camera_index=args.camera,
        confidence=args.confidence,
        mqtt_broker=args.mqtt,
    )

    inference.run(display=not args.no_display, record_path=args.record)


if __name__ == '__main__':
    main()
