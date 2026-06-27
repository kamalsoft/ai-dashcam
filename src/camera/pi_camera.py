# src/camera/pi_camera.py
import os
import time
import threading
import logging
from collections import deque

import cv2
import numpy as np

from src.camera.base_camera import BaseCamera

logger = logging.getLogger("NanovianDashcam")

class PiCamera(BaseCamera):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.writer = None
        self.config = {}
        self.current_metadata = []
        self.latest_annotated_frame = None
        self.frame_width = 640
        self.frame_height = 480
        self.target_fps = 20.0
        self.pre_buffer = deque(maxlen=300)
        self._lock = threading.RLock()

    def initialize(self, config: dict) -> None:
        with self._lock:
            self.config = config
            self.frame_width = int(config.get("frame_width", 640))
            self.frame_height = int(config.get("frame_height", 480))
            self.target_fps = float(config.get("fps", 20.0))

            source = config.get("video_source", 0)
            if isinstance(source, str) and source.isdigit():
                source = int(source)
            elif isinstance(source, str) and source.startswith("/dev/video"):
                try:
                    source = int(source.replace("/dev/video", ""))
                except ValueError:
                    source = 0

            logger.info(f"Connecting to USB Camera index: {source} via V4L2...")
            self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            
            if not self.cap or not self.cap.isOpened():
                raise RuntimeError(f"Could not open V4L2 device index {source}")

            # Establish resolutions
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

            # --- RESILIENT HARDWARE TUNING PROPERTIES ---
            # We attempt to inject camera properties safely without crashing if the sensor doesn't support them
            props = [
                (cv2.CAP_PROP_AUTO_EXPOSURE, 1, "Manual Exposure Mode"), # 1 = Manual Mode
                (cv2.CAP_PROP_EXPOSURE, 150, "Fast Shutter Speed Limit"),
                (cv2.CAP_PROP_CONTRAST, 45, "High Text Contrast Expansion"),
                (cv2.CAP_PROP_GAIN, 24, "Digital Gain Amplification")
            ]

            for prop_id, val, desc in props:
                success = self.cap.set(prop_id, val)
                if success:
                    logger.info(f"  [SUCCESS] Set camera register: {desc} -> {val}")
                else:
                    logger.warning(f"  [SKIPPED] Sensor parameter '{desc}' not native supported by camera firmware.")

            src_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if src_fps > 0 and np.isfinite(src_fps):
                self.target_fps = src_fps

    def _apply_hud(self, frame):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Time: {ts}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {self.target_fps:.1f}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        return frame

    def update_frame(self) -> bool:
        if not self.cap or not self.cap.isOpened():
            return False

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False

        annotated = self._apply_hud(frame.copy())
        
        with self._lock:
            self.latest_annotated_frame = annotated
            self.pre_buffer.append(annotated.copy())

            if self.writer is not None and self.writer.isOpened():
                try:
                    self.writer.write(annotated)
                except Exception:
                    pass

        return True

    def write_pre_buffer_to_incident(self, incident_clip_path: str) -> None:
        os.makedirs(os.path.dirname(incident_clip_path), exist_ok=True)
        with self._lock:
            self.start_recording(incident_clip_path)
            if self.writer is not None and self.writer.isOpened():
                historical_frames = list(self.pre_buffer)
                for f in historical_frames:
                    if f is not None:
                        self.writer.write(f)

    def start_recording(self, output_path: str) -> None:
        with self._lock:
            self.stop_recording()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                output_path,
                fourcc,
                float(self.target_fps),
                (int(self.frame_width), int(self.frame_height)),
            )

    def stop_recording(self) -> None:
        with self._lock:
            if self.writer is not None:
                self.writer.release()
                self.writer = None

    def get_ai_metadata(self) -> list:
        return self.current_metadata

    def get_latest_frame(self):
        with self._lock:
            return self.latest_annotated_frame

    def close(self) -> None:
        with self._lock:
            self.stop_recording()
            if self.cap is not None:
                self.cap.release()
                self.cap = None