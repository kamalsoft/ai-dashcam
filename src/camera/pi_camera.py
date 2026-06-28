import os
import time
import threading
import logging
from collections import deque

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except Exception:  # pragma: no cover
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
        self.frame_width = 1280
        self.frame_height = 720
        self.target_fps = 20.0
        # Rolling historical RAM ring buffer for incident capture tracking
        self.pre_buffer = deque(maxlen=300)
        self._lock = threading.RLock()

    def initialize(self, config: dict) -> None:
        with self._lock:
            self.config = config
            analytics_cfg = config.get("analytics", {})
            
            # Switch default format parameters to clean wide-angle widescreen profiles
            self.frame_width = int(analytics_cfg.get("frame_width", 1280))
            self.frame_height = int(analytics_cfg.get("frame_height", 720))
            self.target_fps = float(analytics_cfg.get("fps", 20.0))
            source = analytics_cfg.get("video_source", "/dev/video0")
            use_picamera2 = bool(analytics_cfg.get("use_picamera2", True))

            self.picam = None
            self.cap = None

            # Primary hardware path: native Raspberry Pi 5 platform engine acceleration
            if use_picamera2 and Picamera2 is not None:
                try:
                    logger.info("Initializing native Raspberry Pi 5 wide-angle hardware layer (Picamera2)...")
                    self.picam = Picamera2()
                    
                    # FIX: Use create_video_configuration with RGB888 to satisfy Pi 5 ISP hardware demands
                    cam_config = self.picam.create_video_configuration(
                        main={"format": "RGB888", "size": (self.frame_width, self.frame_height)},
                        controls={"FrameRate": self.target_fps},
                    )
                    self.picam.configure(cam_config)
                    
                    # Force Auto White Balance tracking parameters to clear blue/yellow distortions
                    self.picam.set_controls({"AwbEnable": True, "AwbMode": 1})
                    
                    self.picam.start()
                    logger.info("Picamera2 hardware pipeline active with native configurations.")
                    return
                except Exception as e:
                    logger.warning("Picamera2 hardware initialization sequence dropped: %s", e)

            # Fallback path: OpenCV capture pipelines
            attempted = []
            candidates = []
            if isinstance(source, str) and "!" in source:
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

            for idx in (0, 1, 2):
                candidates.append((idx, cv2.CAP_V4L2, "fallback-index"))

            for src, backend, label in candidates:
                attempted.append(f"{label}:{src}@{backend}")
                logger.info("Connecting fallback camera source=%s backend=%s (%s)", src, backend, label)
                cap = cv2.VideoCapture(src, backend)
                if cap is not None and cap.isOpened():
                    self.cap = cap
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                    logger.info("OpenCV fallback camera connected successfully.")
                    return
                if cap is not None:
                    cap.release()

            raise RuntimeError("Could not open camera source. Attempted: " + ", ".join(attempted))

    def _apply_hud(self, frame):
        """Applies real-time text telemetry overlays onto the image matrix."""
        pref = self.config.get("user_preferences", {})
        date_fmt = pref.get("date_format", "%Y-%m-%d")
        use_24h = pref.get("time_format_24h", False)
        loc_label = pref.get("custom_location_label", "LYNCUS AI SYSTEM Active")

        time_fmt = "%H:%M:%S" if use_24h else "%I:%M:%S %p"
        full_timestamp_mask = f"{date_fmt} {time_fmt}"
        
        ts = time.strftime(full_timestamp_mask)
        geo_str = f"FPS: {self.target_fps:.1f} | {loc_label}"

        cv2.putText(frame, f"Time: {ts}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, geo_str, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        
        return frame

    def update_frame(self) -> bool:
        raw_frame = None
        if self.picam is not None:
            # Native array from PiSP hardware layer
            raw_frame = self.picam.capture_array()
            # FIX: Invert layout from RGB to clear BGR layout to resolve color shifts
            frame = raw_frame[:, :, ::-1].copy()
        elif self.cap is not None and self.cap.isOpened():
            ok, grabbed = self.cap.read()
            if ok:
                frame = grabbed
            else:
                return False
        else:
            return False

        # Apply regional telemetry text overlays onto the corrected frame matrix
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

            raise RuntimeError(f"Failed to initialize VideoWriter for {output_path}")

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