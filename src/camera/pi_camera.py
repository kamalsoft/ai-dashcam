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
        # 300 frames at 20 FPS maintains a rolling 15-second pre-incident historical RAM ring buffer
        self.pre_buffer = deque(maxlen=300)
        self._lock = threading.RLock()

    def initialize(self, config: dict) -> None:
        with self._lock:
            self.config = config
            analytics_cfg = config.get("analytics", {})
            self.frame_width = int(analytics_cfg.get("frame_width", 640))
            self.frame_height = int(analytics_cfg.get("frame_height", 480))
            self.target_fps = float(analytics_cfg.get("fps", 20.0))

            logger.info("Initializing Raspberry Pi 5 PiSP GStreamer source pipeline...")
            
            # Formulate a native GStreamer ingestion string targeted at the new PiSP layer
            gst_pipeline = (
                f"libcamerasrc ! "
                f"video/x-raw, width={self.frame_width}, height={self.frame_height}, framerate={int(self.target_fps)}/1 ! "
                f"videoconvert ! appsink drop=true max-buffers=2"
            )

            logger.info("Connecting to camera via GStreamer: %s", gst_pipeline)
            
            # Initialize VideoCapture via GStreamer backend explicitly
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

            if self.cap is None or not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera source using Pi5 GStreamer pipeline string: {gst_pipeline}")

            logger.info("Camera connected successfully using libcamerasrc pipeline backend.")

            # --- HARDWARE TUNING NOTE ---
            # Native register parameter modifications (e.g., manual exposure limits) are handled 
            # via rpicam configurations or camera tuning JSON maps inside the Debian OS layer 
            # rather than raw V4L2 ioctl properties on Raspberry Pi 5 hardware.

            src_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if src_fps > 0 and np.isfinite(src_fps):
                self.target_fps = src_fps

    def _apply_hud(self, frame):
        """Applies highly-configurable real-time text layers using regional preferences."""
        pref = self.config.get("user_preferences", {})
        date_fmt = pref.get("date_format", "%Y-%m-%d")
        use_24h = pref.get("time_format_24h", False)
        loc_label = pref.get("custom_location_label", "GPS Monitoring")

        time_fmt = "%H:%M:%S" if use_24h else "%I:%M:%S %p"
        full_timestamp_mask = f"{date_fmt} {time_fmt}"
        
        ts = time.strftime(full_timestamp_mask)
        geo_str = f"FPS: {self.target_fps:.1f} | {loc_label}"

        cv2.putText(frame, f"Time: {ts}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, geo_str, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        
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
                except Exception as e:
                    logger.error("Failed to write frame to active clip: %s", e)

        return True

    def write_pre_buffer_to_incident(self, incident_clip_path: str) -> None:
        """Flushes the rolling pre-buffer safely to disk inside a cross-thread lock block."""
        if not incident_clip_path.endswith(".mkv"):
            incident_clip_path = os.path.splitext(incident_clip_path)[0] + ".mkv"

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
            
            # Force Matroska encapsulation to enforce clean software layout
            if not output_path.endswith(".mkv"):
                output_path = os.path.splitext(output_path)[0] + ".mkv"

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # --- FORCE SOFTWARE LIBX264 COMPRESSION ---
            fourcc = cv2.VideoWriter_fourcc(*"X264")
            self.writer = cv2.VideoWriter(
                output_path,
                fourcc,
                float(self.target_fps),
                (int(self.frame_width), int(self.frame_height))
            )
            logger.info(f"🎥 VideoWriter handle allocated cleanly: {output_path}")

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