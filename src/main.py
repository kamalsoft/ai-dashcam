# src/main.py

import os
import sys
import time
import logging
import threading
import queue
import cv2
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from contextlib import asynccontextmanager

from src.camera.pi_camera import PiCamera
from src.processing.analytics import ThreatAnalytics
from src.storage.circular_buffer import CircularBuffer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger("NanovianDashcam")

APP_CONFIG = {
    "platform": "pi",
    "storage": {
        "clip_dir": str(Path.home() / "ai-dashcam-clips"),
        "max_storage_mb": 2000,
        "clip_duration_seconds": 45,
        "incident_clip_duration_seconds": 120.0,
    },
    "analytics": {
        "min_confidence": 0.40,
        "video_source": "/dev/video0",
        "fps": 20.0,
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
        self.camera = PiCamera()
        self.analytics = ThreatAnalytics(config["analytics"])
        self.storage_manager = CircularBuffer(
            clip_dir=config["storage"]["clip_dir"],
            max_storage_mb=config["storage"]["max_storage_mb"],
        )
        self.active_incident = None
        self.active_normal_clip = None
        self.normal_clip_started_at = 0.0
        self.latest_encoded_frame = None
        
        # Thread-safe queue to pass frames to the ADAS AI processing thread without blocking video recording
        self.ai_frame_queue = queue.Queue(maxsize=2) 
        
        # Late initialization of YOLO model to keep it on its own thread context
        self.yolo_model = None 

    @staticmethod
    def _ts() -> str:
        return time.strftime("%Y%m%d_%H%M%S")

    def _start_normal_clip(self):
        if self.active_incident: return
        clip_root = Path(self.config["storage"]["clip_dir"]) / "normal"
        clip_root.mkdir(parents=True, exist_ok=True)
        clip_path = clip_root / f"clip_{self._ts()}.avi"
        self.camera.start_recording(str(clip_path))
        self.storage_manager.register_active_file(str(clip_path))
        self.active_normal_clip = str(clip_path)
        self.normal_clip_started_at = time.monotonic()
        logger.info(f"🟢 Continuous Loop Recording started: {clip_path.name}")

    def trigger_incident_containment(self, frame):
        """Halts standard loop recording and moves historical buffers safely to an incident file."""
        if self.active_incident:
            return  # Already handling an active incident

        logger.warning("🚨 ADAS THREAT DETECTED! Initiating Incident Lockout Context...")
        
        if self.active_normal_clip:
            self.camera.stop_recording()
            self.storage_manager.unregister_active_file(self.active_normal_clip)
            self.active_normal_clip = None

        incident_root = Path(self.config["storage"]["clip_dir"]) / "incidents" / f"incident_{self._ts()}"
        incident_root.mkdir(parents=True, exist_ok=True)

        snapshot_path = incident_root / "snapshot.jpg"
        clip_path = incident_root / "incident_footage.avi"

        if frame is not None:
            cv2.imwrite(str(snapshot_path), frame)

        # Flush pre-buffer (historical frames) to the incident file
        self.camera.write_pre_buffer_to_incident(str(clip_path))
        self.storage_manager.register_active_file(str(clip_path))

        self.active_incident = {
            "root_dir": str(incident_root),
            "clip_path": str(clip_path),
            "started_at": time.monotonic()
        }

    def _adas_worker_loop(self):
        """Asynchronous worker loop dedicated entirely to heavy ML inference tasks."""
        from ultralytics import YOLO
        logger.info("Loading YOLOv8 Model onto ADAS Context Thread...")
        self.yolo_model = YOLO("yolov8n.pt")
        logger.info("YOLOv8 Model loaded successfully. ADAS actively scanning.")

        while self.is_running:
            try:
                # Grab the latest frame dropped by the camera stream loop
                frame = self.ai_frame_queue.get(timeout=1.0)
                
                # Run YOLO tracking with persistence to assign consistent tracking IDs to vehicles
                results = self.yolo_model.track(frame, persist=True, verbose=False)
                
                if results and len(results) > 0:
                    boxes_data = results[0].boxes
                    parsed_metadata = []
                    
                    if boxes_data is not None and boxes_data.id is not None:
                        xyxy = boxes_data.xyxy.cpu().numpy()
                        conf = boxes_data.conf.cpu().numpy()
                        cls = boxes_data.cls.cpu().numpy()
                        track_ids = boxes_data.id.cpu().numpy()
                        
                        for i in range(len(track_ids)):
                            parsed_metadata.append({
                                "box": xyxy[i].tolist(),
                                "conf": float(conf[i]),
                                "cls": int(cls[i]),
                                "track_id": int(track_ids[i])
                            })
                    
                    # Pass tracking metadata to your expansion metrics processing logic
                    is_threat = self.analytics.process_inference_metadata(parsed_metadata)
                    if is_threat:
                        self.trigger_incident_containment(frame)
                        
                    # Render the visual bounding boxes on the live web stream dashboard
                    annotated_frame = results[0].plot()
                    ret, encoded_img = cv2.imencode('.jpg', annotated_frame)
                    if ret:
                        self.latest_encoded_frame = encoded_img.tobytes()
                        
                self.ai_frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error inside ADAS worker engine: {e}")

    def run_lifecycle(self):
        """High-frequency thread for continuous, steady 20 FPS video recording."""
        self.camera.initialize(self.config["analytics"])
        
        # Start the background ADAS AI processing worker thread
        threading.Thread(target=self._adas_worker_loop, daemon=True).start()
        
        self._start_normal_clip()
        frame_duration = 1.0 / self.config["analytics"]["fps"]

        try:
            while self.is_running:
                loop_start = time.monotonic()
                self.storage_manager.enforce_retention_policy_async(".avi")

                # Handle incident recording duration timeouts
                if self.active_incident:
                    elapsed = time.monotonic() - self.active_incident["started_at"]
                    if elapsed >= self.config["storage"]["incident_clip_duration_seconds"]:
                        logger.info("Incident window expired. Resuming normal loop operations.")
                        self.camera.stop_recording()
                        self.storage_manager.unregister_active_file(self.active_incident["clip_path"])
                        self.active_incident = None
                        self._start_normal_clip()
                else:
                    # Handle normal clip rotation
                    if (time.monotonic() - self.normal_clip_started_at) >= self.config["storage"]["clip_duration_seconds"]:
                        self.camera.stop_recording()
                        self.storage_manager.unregister_active_file(self.active_normal_clip)
                        self.active_normal_clip = None
                        self._start_normal_clip()

                ok = self.camera.update_frame()
                if not ok: break

                raw_frame = self.camera.get_latest_frame()
                
                # Push the raw frame to the ADAS processing queue if space is available
                if raw_frame is not None:
                    try:
                        self.ai_frame_queue.put_nowait(raw_frame)
                    except queue.Full:
                        pass # Drop frames if the AI thread is running behind to keep video recording smooth
                    
                    # If the AI thread hasn't updated the live stream frame yet, fall back to the raw frame
                    if self.latest_encoded_frame is None:
                        ret, encoded_img = cv2.imencode('.jpg', raw_frame)
                        if ret: self.latest_encoded_frame = encoded_img.tobytes()

                sleep_for = frame_duration - (time.monotonic() - loop_start)
                if sleep_for > 0: time.sleep(sleep_for)
        finally:
            self.camera.close()

# FastAPI Server Context Routing Config Layer
@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=orchestrator.run_lifecycle, daemon=True).start()
    yield
    orchestrator.is_running = False

app = FastAPI(title="Nanovian AI Dashcam Gateway", lifespan=lifespan)
orchestrator = DashcamOrchestrator(APP_CONFIG)

# Mount the storage directory so files can be requested over HTTP
Path(APP_CONFIG["storage"]["clip_dir"]).mkdir(parents=True, exist_ok=True)
app.mount("/clips", StaticFiles(directory=APP_CONFIG["storage"]["clip_dir"]), name="clips")

@app.get("/video_feed")
async def video_feed():
    async def frame_generator():
        while orchestrator.is_running:
            if orchestrator.latest_encoded_frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + orchestrator.latest_encoded_frame + b'\r\n')
            await asyncio.sleep(0.05)
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/list_incidents")
def list_incidents():
    """Returns a JSON catalog of all protected incident files available for playback."""
    incidents_dir = Path(APP_CONFIG["storage"]["clip_dir"]) / "incidents"
    if not incidents_dir.exists():
        return {"incidents": []}
    
    entries = []
    for p in sorted(incidents_dir.iterdir(), reverse=True):
        if p.is_dir():
            entries.append({
                "incident_id": p.name,
                "has_snapshot": (p / "snapshot.jpg").exists(),
                "video_url": f"/clips/incidents/{p.name}/incident_footage.avi",
                "snapshot_url": f"/clips/incidents/{p.name}/snapshot.jpg" if (p / "snapshot.jpg").exists() else None
            })
    return {"incidents": entries}

if __name__ == "__main__":
    uvicorn.run("src.main:app", host=APP_CONFIG["network"]["bind_address"], port=APP_CONFIG["network"]["port"], workers=1)