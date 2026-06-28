# src/camera/pi_camera.py
import os
import time
import threading
import logging
from collections import deque

import cv2
import numpy as np

# Import the native Raspberry Pi camera engine
try:
    from picamera2 import Picamera2
except Exception:  # pragma: no cover - runtime dependency on Pi images
    Picamera2 = None

from src.camera.base_camera import BaseCamera

logger = logging.getLogger("NanovianDashcam")

class PiCamera(BaseCamera):
    def __init__(self):
        super().__init__()
        self.picam = None
        self.cap = None
        self.writer = None
        self.config = {}
        self.current_metadata = []
        self.latest_annotated_frame = None
        self.frame_width = 640
        self.frame_height = 480
        self.target_fps = 20.0
        # 300 frames maintains a rolling 15-second pre-incident historical RAM ring buffer
        self.pre_buffer = deque(maxlen=300)
        self._lock = threading.RLock()

    def initialize(self, config: dict) -> None:
        with self._lock:
            self.config = config
            analytics_cfg = config.get("analytics", {})
            self.frame_width = int(analytics_cfg.get("frame_width", 640))
            self.frame_height = int(analytics_cfg.get("frame_height", 480))
            self.target_fps = float(analytics_cfg.get("fps", 20.0))
            source = analytics_cfg.get("video_source", "/dev/video0")
            use_picamera2 = bool(analytics_cfg.get("use_picamera2", True))

            self.picam = None
            self.cap = None

            # Primary path: native Pi camera stack.
            if use_picamera2 and Picamera2 is not None:
                try:
                    logger.info("Initializing native Raspberry Pi camera layer (Picamera2)...")
                    self.picam = Picamera2()
                    cam_config = self.picam.create_preview_configuration(
                        main={"format": "BGR24", "size": (self.frame_width, self.frame_height)},
                        controls={"FrameRate": self.target_fps},
                    )
                    self.picam.configure(cam_config)
                    self.picam.start()
                    logger.info("Picamera2 connected successfully.")
                    return
                except Exception as e:
                    logger.warning("Picamera2 initialization failed: %s", e)

            # Fallback path: OpenCV capture (supports GStreamer pipelines and V4L2 indexes).
            attempted = []
            candidates = []
            if isinstance(source, str) and "!" in source:
                # Treat string as a GStreamer pipeline.
                candidates.append((source, cv2.CAP_GSTREAMER, "gstreamer-pipeline"))
                candidates.append((source, cv2.CAP_ANY, "generic-pipeline"))
            elif isinstance(source, str) and source.startswith("/dev/video"):
                candidates.append((source, cv2.CAP_V4L2, "v4l2-device"))
                try:
                    idx = int(source.replace("/dev/video", ""))
                    candidates.append((idx, cv2.CAP_V4L2, "v4l2-index"))
                except ValueError:
                    pass
            elif isinstance(source, str) and source.isdigit():
                candidates.append((int(source), cv2.CAP_V4L2, "v4l2-index"))
            else:
                candidates.append((source, cv2.CAP_V4L2, "source"))

            # Common camera indexes as final fallback.
            for idx in (0, 1, 2):
                candidates.append((idx, cv2.CAP_V4L2, "fallback-index"))
                candidates.append((idx, cv2.CAP_ANY, "fallback-any"))

            for src, backend, label in candidates:
                attempted.append(f"{label}:{src}@{backend}")
                logger.info("Connecting to camera source=%s backend=%s (%s)", src, backend, label)
                cap = cv2.VideoCapture(src, backend)
                if cap is not None and cap.isOpened():
                    self.cap = cap
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                    logger.info("OpenCV camera connected successfully using source=%s backend=%s", src, backend)
                    return
                if cap is not None:
                    cap.release()

            raise RuntimeError("Could not open camera source. Attempted: " + ", ".join(attempted))

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
        frame = None
        if self.picam is not None:
            frame = self.picam.capture_array()
        elif self.cap is not None and self.cap.isOpened():
            ok, grabbed = self.cap.read()
            if ok:
                frame = grabbed

        if frame is None:
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
            
            if not output_path.endswith(".mkv"):
                output_path = os.path.splitext(output_path)[0] + ".mkv"

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            fps = float(self.target_fps)
            frame_size = (int(self.frame_width), int(self.frame_height))
            codec_candidates = ["MJPG", "XVID", "mp4v"]

            self.writer = None
            for codec in codec_candidates:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
                if writer is not None and writer.isOpened():
                    self.writer = writer
                    logger.info("VideoWriter initialized: codec=%s path=%s", codec, output_path)
                    return
                if writer is not None:
                    writer.release()

            raise RuntimeError(
                "Failed to initialize VideoWriter for "
                f"{output_path}. Tried codecs: {', '.join(codec_candidates)}"
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
            if self.picam is not None:
                self.picam.stop()
                self.picam = None
            if self.cap is not None:
                self.cap.release()
                self.cap = None