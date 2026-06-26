# src/main.py
import os
import sys
import multiprocessing

# Must be configured before heavy imports
if sys.platform == "darwin":
    try:
        # Use spawn on macOS to avoid fork/atfork logging issues
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["YOLO_VERBOSE"] = "False"

import time
import logging
import threading
import http.server
import socketserver
import cv2
import signal
import traceback
from pathlib import Path

from src.camera.mac_camera import MacCamera
from src.camera.pi_camera import PiCamera
from src.processing.analytics import ThreatAnalytics
from src.storage.circular_buffer import CircularBuffer

# Configure structured system logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger("DashcamCore")

# Configuration Block
APP_CONFIG = {
    "platform": "mac",
    "storage": {
        "clip_dir": "./mock_dashcam_clips",
        "max_storage_mb": 1000,
        "clip_duration_seconds": 45,
        "incident_postbuffer_seconds": 2.0,
        "incident_clip_duration_seconds": 120.0,  # 2 minutes
    },
    "analytics": {
        "min_confidence": 0.45,
        "video_source": "./assets/test_dashcam.mp4",
        "fps": 20.0,
        "preview": True,
        "preview_window_name": "AI Dashcam Preview",
    },
    "network": {
        "bind_address": "0.0.0.0",
        "port": 8080,
    },
}

class DashcamOrchestrator:
    def __init__(self, config: dict):
        self.config = config
        self.is_running = True
        self.shutdown_event = threading.Event()
        self.camera = MacCamera() if config.get("platform", "mac") == "mac" else PiCamera()
        self.analytics = ThreatAnalytics(config.get("analytics", {}))
        self.storage_manager = CircularBuffer(
            clip_dir=config["storage"]["clip_dir"],
            max_storage_mb=config["storage"]["max_storage_mb"],
        )
        self.active_incident = None
        self.active_normal_clip = None
        self.normal_clip_started_at = 0.0
        self.postbuffer_seconds = float(config["storage"].get("incident_postbuffer_seconds", 2.0))
        self.clip_duration_seconds = float(config["storage"].get("clip_duration_seconds", 45))
        self.incident_clip_duration_seconds = float(
            config["storage"].get("incident_clip_duration_seconds", 120.0)
        )
        self._install_signal_handlers()  # <-- was missing

    def _install_signal_handlers(self):
        def _handler(signum, _frame):
            self.is_running = False
            self.shutdown_event.set()
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    @staticmethod
    def _ts() -> str:
        return time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"

    def _start_normal_clip(self):
        clip_root = Path(self.config["storage"]["clip_dir"])
        clip_root.mkdir(parents=True, exist_ok=True)

        ts = self._ts()
        clip_path = clip_root / f"clip_{ts}.avi"

        self.camera.start_recording(str(clip_path))
        self.storage_manager.register_active_file(str(clip_path))
        self.active_normal_clip = str(clip_path)
        self.normal_clip_started_at = time.monotonic()
        logger.info("Normal recording started: %s", clip_path.name)

    def _rotate_normal_clip_if_needed(self):
        if self.active_normal_clip is None:
            self._start_normal_clip()
            return

        if (time.monotonic() - self.normal_clip_started_at) >= self.clip_duration_seconds:
            self.camera.stop_recording()
            self.storage_manager.unregister_active_file(self.active_normal_clip)
            self.active_normal_clip = None
            self._start_normal_clip()

    def _start_incident(self, frame):
        # pause normal writer
        if self.active_normal_clip is not None:
            self.camera.stop_recording()
            self.storage_manager.unregister_active_file(self.active_normal_clip)
            self.active_normal_clip = None

        ts = self._ts()
        incident_dir = Path(self.config["storage"]["clip_dir"]) / f"incident_{ts}"
        incident_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = incident_dir / f"snapshot_{ts}.jpg"
        incident_clip_path = incident_dir / f"clip_{ts}.avi"

        cv2.imwrite(str(snapshot_path), frame)

        self.storage_manager.register_active_file(str(incident_clip_path))
        self.camera.write_pre_buffer_to_incident(str(incident_clip_path))

        self.active_incident = {
            "clip_path": str(incident_clip_path),
            "last_seen": time.monotonic(),
            "started_at": time.monotonic(),  # fixed-duration anchor
        }
        logger.info("Incident recording started: %s", incident_clip_path)

    def _stop_incident_and_resume_normal(self):
        if self.active_incident is None:
            return

        self.camera.stop_recording()
        self.storage_manager.unregister_active_file(self.active_incident["clip_path"])
        logger.info("Incident recording stopped.")
        self.active_incident = None

        # resume normal recording in root clip directory
        self._start_normal_clip()

    def run_lifecycle(self):
        self.camera.initialize(self.config["analytics"])
        target_fps = float(getattr(self.camera, "target_fps", 20.0) or 20.0)
        frame_duration = 1.0 / max(target_fps, 1.0)

        preview_enabled = bool(self.config["analytics"].get("preview", True))
        preview_window = self.config["analytics"].get("preview_window_name", "AI Dashcam Preview")
        if preview_enabled:
            cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)

        self._start_normal_clip()

        try:
            while self.is_running and not self.shutdown_event.is_set():
                loop_start = time.monotonic()
                self.storage_manager.enforce_retention_policy_async(".avi")

                if self.active_incident is None:
                    self._rotate_normal_clip_if_needed()

                ok = self.camera.update_frame()
                if not ok:
                    self.is_running = False
                    break

                frame = self.camera.get_latest_frame()
                metadata = self.camera.get_ai_metadata()
                threat = self.analytics.process_inference_metadata(metadata)

                if threat and self.active_incident is None and frame is not None:
                    self._start_incident(frame)
                elif threat and self.active_incident is not None:
                    self.active_incident["last_seen"] = time.monotonic()

                if self.active_incident is not None:
                    incident_elapsed = time.monotonic() - self.active_incident["started_at"]
                    if incident_elapsed >= self.incident_clip_duration_seconds:
                        self._stop_incident_and_resume_normal()

                if preview_enabled and frame is not None:
                    cv2.imshow(preview_window, frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        self.is_running = False
                        break

                elapsed = time.monotonic() - loop_start
                sleep_for = frame_duration - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            if self.active_incident is not None:
                self._stop_incident_and_resume_normal()
            if self.active_normal_clip is not None:
                self.camera.stop_recording()
                self.storage_manager.unregister_active_file(self.active_normal_clip)
                self.active_normal_clip = None
            if preview_enabled:
                cv2.destroyWindow(preview_window)
            self._shutdown()

    def _shutdown(self):
        self.shutdown_event.set()
        try:
            self.camera.stop_recording()
        finally:
            if self.active_incident is not None:
                self.storage_manager.unregister_active_file(self.active_incident["clip_path"])
                self.active_incident = None
            self.camera.close()

def main() -> int:
    logger.info("Starting AI Dashcam...")
    orchestrator = DashcamOrchestrator(APP_CONFIG)
    try:
        orchestrator.run_lifecycle()
        logger.info("AI Dashcam stopped cleanly.")
        return 0
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down cleanly.")
        return 0
    except Exception:
        logger.exception("Fatal error in dashcam runtime")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())