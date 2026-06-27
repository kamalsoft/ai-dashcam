# src/main.py
import os
import sys
import multiprocessing

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["YOLO_VERBOSE"] = "False"

import time
import logging
import threading
import cv2
import signal
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

from src.camera.pi_camera import PiCamera
from src.processing.analytics import ThreatAnalytics
from src.storage.circular_buffer import CircularBuffer

# Configure structured system logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger("DashcamCore")

# Hardened Configuration Block for Raspberry Pi 5 Headless Deployment
APP_CONFIG = {
    "platform": "pi",  # Swapped to Pi natively
    "storage": {
        "clip_dir": "./mock_dashcam_clips",
        "max_storage_mb": 1000,
        "clip_duration_seconds": 45,
        "incident_postbuffer_seconds": 2.0,
        "incident_clip_duration_seconds": 120.0,
    },
    "analytics": {
        "min_confidence": 0.45,
        "video_source": "/dev/video0",  # Pointing to your Winsafe device
        "fps": 20.0,
        "preview": False,               # Disabled cv2.imshow GUI dependencies
        "preview_window_name": "AI Dashcam Preview",
    },
    "network": {
        "bind_address": "0.0.0.0",
        "port": 8000,
    },
}

class DashcamOrchestrator:
    def __init__(self, config: dict):
        self.config = config
        self.is_running = True
        self.shutdown_event = threading.Event()
        self.camera = PiCamera()  # Hard-targeted Pi processing engine
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
        self.latest_encoded_frame = None  # Global frame cache pointer for web streaming
        self._install_signal_handlers()

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
            "started_at": time.monotonic(),
        }
        logger.info("Incident recording started: %s", incident_clip_path)

    def _stop_incident_and_resume_normal(self):
        if self.active_incident is None:
            return
        self.camera.stop_recording()
        self.storage_manager.unregister_active_file(self.active_incident["clip_path"])
        logger.info("Incident recording stopped.")
        self.active_incident = None
        self._start_normal_clip()

    def run_lifecycle(self):
        """Executes inside a dedicated background thread to keep video ingestion fluid."""
        self.camera.initialize(self.config["analytics"])
        target_fps = float(getattr(self.camera, "target_fps", 20.0) or 20.0)
        frame_duration = 1.0 / max(target_fps, 1.0)

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

                # Update the stream pointer with a compressed MJPEG matrix for FastAPI
                if frame is not None:
                    ret, encoded_img = cv2.imencode('.jpg', frame)
                    if ret:
                        self.latest_encoded_frame = encoded_img.tobytes()

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

# Initialize API Server and Orchestrator
orchestrator = DashcamOrchestrator(APP_CONFIG)
orchestrator_thread = None

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start and stop the orchestrator thread using FastAPI lifespan events."""
    global orchestrator_thread
    orchestrator_thread = threading.Thread(target=orchestrator.run_lifecycle, daemon=True)
    orchestrator_thread.start()
    try:
        yield
    finally:
        orchestrator.is_running = False
        orchestrator.shutdown_event.set()
        if orchestrator_thread is not None:
            orchestrator_thread.join(timeout=5)

app = FastAPI(title="Nanovian AI Dashcam Gateway", lifespan=lifespan)

async def frame_generator():
    """Asynchronously streams the active memory-matrix to network ports."""
    while orchestrator.is_running:
        if orchestrator.latest_encoded_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + orchestrator.latest_encoded_frame + b'\r\n')
        await asyncio.sleep(0.04)  # ~25 FPS throttling to minimize CPU churn

@app.get("/video_feed")
async def video_feed():
    """Exposes real-time vehicle viewing link to web browsers."""
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
def get_status():
    """Returns runtime state variables for local analytics tracking."""
    return {
        "system_active": orchestrator.is_running,
        "incident_active": orchestrator.active_incident is not None,
        "normal_clip": orchestrator.active_normal_clip
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=APP_CONFIG["network"]["bind_address"], 
        port=APP_CONFIG["network"]["port"], 
        workers=1
    )
