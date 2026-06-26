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
        self.pre_buffer = deque(maxlen=300)

    def initialize(self, config: dict) -> None:
        self.config = config
        self.frame_width = int(config.get("frame_width", 640))
        self.frame_height = int(config.get("frame_height", 480))
        self.target_fps = float(config.get("fps", 20.0))

        source = config.get("video_source", 0)
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Pi camera initialization failed.")

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
        self.latest_annotated_frame = annotated
        self.pre_buffer.append(annotated.copy())

        if self.writer is not None:
            self.writer.write(annotated)

        return True

    def write_pre_buffer_to_incident(self, incident_clip_path: str) -> None:
        os.makedirs(os.path.dirname(incident_clip_path), exist_ok=True)
        self.start_recording(incident_clip_path)
        for f in self.pre_buffer:
            self.writer.write(f)

    def start_recording(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
        cv2.destroyAllWindows()