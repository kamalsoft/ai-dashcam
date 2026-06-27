# src/camera/pi_camera.py

import os
import time
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
        # 300 frames at 20fps gives exactly 15 seconds of rolling pre-incident memory safety
        self.pre_buffer = deque(maxlen=300)

    def initialize(self, config: dict) -> None:
        self.config = config
        self.frame_width = int(config.get("frame_width", 640))
        self.frame_height = int(config.get("frame_height", 480))
        self.target_fps = float(config.get("fps", 20.0))

        # Capture video_source safely. Defaults to /dev/video0
        source = config.get("video_source", 0)
        
        # If source is a string representing a digit, convert it to an integer for OpenCV index routing
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Pi camera initialization failed for source: {source}")

        src_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_fps > 0 and np.isfinite(src_fps):
            self.target_fps = src_fps

    def _apply_hud(self, frame):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        # Anti-aliased high contrast cyan text overlay
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
        self.latest_annotated_frame = annotated
        self.pre_buffer.append(annotated.copy())

        # If a video chunk file is actively open, commit the frame array directly
        if self.writer is not None and self.writer.isOpened():
            self.writer.write(annotated)

        return True

    def write_pre_buffer_to_incident(self, incident_clip_path: str) -> None:
        """Safely flushes the internal circular RAM ring straight to disk storage."""
        os.makedirs(os.path.dirname(incident_clip_path), exist_ok=True)
        
        # Instantiates the underlying stream container engine
        self.start_recording(incident_clip_path)
        
        if self.writer is not None and self.writer.isOpened():
            # Flush existing historical pre-buffer data sequentially
            for f in list(self.pre_buffer):
                if f is not None:
                    self.writer.write(f)

    def start_recording(self, output_path: str) -> None:
        self.stop_recording()  # Hard assurance to close prior file bindings before reallocating pointers
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Standard AVI compression profile optimized for ARM processing layouts
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            float(self.target_fps),
            (int(self.frame_width), int(self.frame_height)),
        )

    def stop_recording(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def get_ai_metadata(self) -> list:
        return self.current_metadata

    def get_latest_frame(self):
        return self.latest_annotated_frame

    def close(self) -> None:
        self.stop_recording()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
