# src/camera/pi_camera.py
import os
import time
import threading
from collections import deque

import cv2
import numpy as np

from src.camera.base_camera import BaseCamera

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
        # 300 frames at 20 FPS provides a rolling 15-second pre-incident memory buffer
        self.pre_buffer = deque(maxlen=300)
        
        # Reentrant lock protects VideoWriter bindings from concurrent multi-threaded access
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
                
            self.cap = cv2.VideoCapture(source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

            if not self.cap or not self.cap.isOpened():
                raise RuntimeError(f"Pi camera hardware initialization failed for source: {source}")

            # Optimize V4L2 drivers to eliminate motion blur and prevent overexposed license plates
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = Manual Exposure Mode
            self.cap.set(cv2.CAP_PROP_EXPOSURE, 150)     # Fast shutter value limits daytime/nighttime lens blur
            self.cap.set(cv2.CAP_PROP_CONTRAST, 45)      # Enhanced edge contrast helps isolate license plate characters
            self.cap.set(cv2.CAP_PROP_GAIN, 24)          # Sensor gain handles illumination recovery

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
        """Safely locks camera resources while flushing rolling RAM history to disk."""
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
            
            # Using mp4v container for maximum structural stability on ARM64 platforms
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