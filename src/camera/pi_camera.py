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

            source = analytics_cfg.get("video_source", 0)
            candidates = []
            if isinstance(source, str) and source.isdigit():
                candidates.append(int(source))
            elif isinstance(source, str) and source.startswith("/dev/video"):
                candidates.append(source)
                try:
                    candidates.append(int(source.replace("/dev/video", "")))
                except ValueError:
                    pass
            else:
                candidates.append(source)

            if 0 not in candidates: candidates.append(0)
            if 1 not in candidates: candidates.append(1)
            if 2 not in candidates: candidates.append(2)

            self.cap = None
            opened_from = None
            attempted = []
            
            for candidate in candidates:
                for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
                    attempted.append(f"{candidate} via backend {backend}")
                    logger.info("Connecting to camera source: %s (backend=%s)", candidate, backend)
                    cap = cv2.VideoCapture(candidate, backend)
                    if cap is not None and cap.isOpened():
                        self.cap = cap
                        opened_from = (candidate, backend)
                        break
                    if cap is not None:
                        cap.release()
                if self.cap is not None:
                    break

            if self.cap is None:
                raise RuntimeError("Could not open camera source. Attempted: " + ", ".join(attempted))

            logger.info("Camera connected successfully using source=%s backend=%s", opened_from[0], opened_from[1])

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

            # --- RESILIENT V4L2 HARDWARE TUNING PROPERTIES ---
            props = [
                (cv2.CAP_PROP_AUTO_EXPOSURE, 1, "Manual Exposure Mode"),
                (cv2.CAP_PROP_EXPOSURE, 150, "Fast Shutter Speed Limit"),
                (cv2.CAP_PROP_CONTRAST, 45, "High Text Contrast Expansion"),
                (cv2.CAP_PROP_GAIN, 24, "Digital Gain Amplification")
            ]

            for prop_id, val, desc in props:
                success = self.cap.set(prop_id, val)
                if success:
                    logger.info(f"  [SUCCESS] Set camera register: {desc} -> {val}")
                else:
                    logger.warning(f"  [SKIPPED] Sensor parameter '{desc}' not natively supported by camera firmware.")

            src_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if src_fps > 0 and np.isfinite(src_fps):
                self.target_fps = src_fps

    def _apply_hud(self, frame):
        """Applies highly-configurable real-time text layers using regional preferences."""
        pref = self.config.get("user_preferences", {})
        gps = self.config.get("gps", {})
        date_fmt = pref.get("date_format", "%Y-%m-%d")
        use_24h = pref.get("time_format_24h", False)
        loc_label = pref.get("custom_location_label", "GPS Monitoring")

        time_fmt = "%H:%M:%S" if use_24h else "%I:%M:%S %p"
        full_timestamp_mask = f"{date_fmt} {time_fmt}"
        
        ts = time.strftime(full_timestamp_mask)
        geo_str = f"FPS: {self.target_fps:.1f} | {loc_label}"
        lat = gps.get("latitude")
        lon = gps.get("longitude")

        cv2.putText(frame, f"Time: {ts}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, geo_str, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        if lat is not None and lon is not None:
            try:
                lat_val = float(lat)
                lon_val = float(lon)
                cv2.putText(
                    frame,
                    f"Lat: {lat_val:.6f}  Lon: {lon_val:.6f}",
                    (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            except (TypeError, ValueError):
                pass
        
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