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
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

from src.camera.pi_camera import PiCamera
from src.processing.analytics import ThreatAnalytics
from src.storage.circular_buffer import CircularBuffer

# Initialize logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger("NanovianDashcam")

APP_CONFIG = {
    "storage": {
        "clip_dir": str(Path.home() / "ai-dashcam-clips"),
        "max_storage_mb": 2000,
        "clip_duration_seconds": 45,
        "incident_clip_duration_seconds": 120,
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
    "user_preferences": {
        "timezone": "America/Chicago",
        "time_format_24h": False,           # False = 12-Hour Clock with AM/PM
        "date_format": "%Y-%m-%d",          # Standard ISO-Style notation
        "custom_location_label": "Naperville, IL",
    }
}

class UserPreferencesSchema(BaseModel):
    custom_location_label: str
    time_format_24h: bool
    date_format: str

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
        self.ai_frame_queue = queue.Queue(maxsize=2)
        self.yolo_model = None

    @staticmethod
    def _ts() -> str:
        return time.strftime("%Y%m%d_%H%M%S")

    def _start_normal_clip(self):
        if self.active_incident: 
            return
        clip_root = Path(self.config["storage"]["clip_dir"]) / "normal"
        clip_root.mkdir(parents=True, exist_ok=True)
        clip_path = clip_root / f"clip_{self._ts()}.mp4"
        self.camera.start_recording(str(clip_path))
        self.storage_manager.register_active_file(str(clip_path))
        self.active_normal_clip = str(clip_path)
        self.normal_clip_started_at = time.monotonic()
        logger.info(f"🟢 Continuous Loop Recording started: {clip_path.name}")

    def trigger_incident_containment(self, frame):
        if self.active_incident:
            return

        logger.warning("🚨 ADAS THREAT VERIFIED! Initiating Locked Incident Containment...")
        if self.active_normal_clip:
            self.camera.stop_recording()
            self.storage_manager.unregister_active_file(self.active_normal_clip)
            self.active_normal_clip = None

        incident_root = Path(self.config["storage"]["clip_dir"]) / "incidents" / f"incident_{self._ts()}"
        incident_root.mkdir(parents=True, exist_ok=True)

        snapshot_path = incident_root / "snapshot.jpg"
        clip_path = incident_root / "incident_footage.mp4"

        if frame is not None:
            cv2.imwrite(str(snapshot_path), frame)

        self.camera.write_pre_buffer_to_incident(str(clip_path))
        self.storage_manager.register_active_file(str(clip_path))

        self.active_incident = {
            "root_dir": str(incident_root),
            "clip_path": str(clip_path),
            "started_at": time.monotonic()
        }

    def _adas_worker_loop(self):
        """Isolated background context thread dedicated to running ML model inference."""
        from ultralytics import YOLO
        logger.info("Loading YOLOv8 Model onto ADAS Context Thread...")
        self.yolo_model = YOLO("yolov8n.pt")
        logger.info("YOLOv8 Model loaded successfully. ADAS tracking active.")

        while self.is_running:
            try:
                frame = self.ai_frame_queue.get(timeout=1.0)
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
                    
                    is_threat = self.analytics.process_inference_metadata(parsed_metadata)
                    if is_threat:
                        self.trigger_incident_containment(frame)
                        
                    annotated_frame = frame.copy()
                    for det in parsed_metadata:
                        bx = det["box"]
                        speed_val = det.get("speed_mph", 0.0)
                        tid = det["track_id"]
                        
                        cv2.rectangle(annotated_frame, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0, 255, 0), 2)
                        speed_label = f"ID: {tid} | {abs(speed_val):.1f} MPH"
                        cv2.putText(
                            annotated_frame, speed_label, (int(bx[0]), int(bx[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA
                        )
                    
                    ret, encoded_img = cv2.imencode('.jpg', annotated_frame)
                    if ret:
                        self.latest_encoded_frame = encoded_img.tobytes()
                        
                self.ai_frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error inside ADAS worker engine: {e}")

    def run_lifecycle(self):
        """High-priority loop ensuring steady camera frame capture and video encoding."""
        try:
            self.camera.initialize(self.config)
        except Exception as e:
            logger.error("Camera initialization failed: %s", e)
            self.is_running = False
            return

        threading.Thread(target=self._adas_worker_loop, daemon=True).start()
        self._start_normal_clip()
        
        frame_duration = 1.0 / self.config["analytics"]["fps"]

        try:
            while self.is_running:
                loop_start = time.monotonic()
                self.storage_manager.enforce_retention_policy_async(".mp4")

                if self.active_incident:
                    elapsed = time.monotonic() - self.active_incident["started_at"]
                    if elapsed >= self.config["storage"]["incident_clip_duration_seconds"]:
                        logger.info("Incident window expired. Resuming standard loop operations.")
                        self.camera.stop_recording()
                        self.storage_manager.unregister_active_file(self.active_incident["clip_path"])
                        self.active_incident = None
                        self._start_normal_clip()
                else:
                    if (time.monotonic() - self.normal_clip_started_at) >= self.config["storage"]["clip_duration_seconds"]:
                        self.camera.stop_recording()
                        self.storage_manager.unregister_active_file(self.active_normal_clip)
                        self.active_normal_clip = None
                        self._start_normal_clip()

                ok = self.camera.update_frame()
                if not ok: 
                    break

                raw_frame = self.camera.get_latest_frame()
                if raw_frame is not None:
                    try:
                        self.ai_frame_queue.put_nowait(raw_frame)
                    except queue.Full:
                        pass
                    
                    if self.latest_encoded_frame is None:
                        ret, encoded_img = cv2.imencode('.jpg', raw_frame)
                        if ret: 
                            self.latest_encoded_frame = encoded_img.tobytes()

                sleep_for = frame_duration - (time.monotonic() - loop_start)
                if sleep_for > 0: 
                    time.sleep(sleep_for)
        finally:
            self.camera.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=orchestrator.run_lifecycle, daemon=True).start()
    yield
    orchestrator.is_running = False

app = FastAPI(title="Nanovian AI Dashcam Gateway", lifespan=lifespan)
orchestrator = DashcamOrchestrator(APP_CONFIG)

# Mount root clip storage directory for static layout delivery
Path(APP_CONFIG["storage"]["clip_dir"]).mkdir(parents=True, exist_ok=True)
app.mount("/clips", StaticFiles(directory=APP_CONFIG["storage"]["clip_dir"]), name="clips")

@app.get("/", response_class=HTMLResponse)
def serve_dashboard_home_screen():
    """Serves the Single Page Application UI control dashboard framework."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Web interface HTML template asset missing from source tree.")
    return template_path.read_text()

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
    incidents_dir = Path(APP_CONFIG["storage"]["clip_dir"]) / "incidents"
    if not incidents_dir.exists():
        return {"incidents": []}
    
    entries = []
    for p in sorted(incidents_dir.iterdir(), reverse=True):
        if p.is_dir():
            entries.append({
                "incident_id": p.name,
                "has_snapshot": (p / "snapshot.jpg").exists(),
                "video_url": f"/clips/incidents/{p.name}/incident_footage.mp4",
                "snapshot_url": f"/clips/incidents/{p.name}/snapshot.jpg" if (p / "snapshot.jpg").exists() else None
            })
    return {"incidents": entries}

@app.post("/update_preferences")
def update_preferences(prefs: UserPreferencesSchema):
    """Hot-reloads user regional parameters into app memory on-the-fly."""
    try:
        APP_CONFIG["user_preferences"]["custom_location_label"] = prefs.custom_location_label
        APP_CONFIG["user_preferences"]["time_format_24h"] = prefs.time_format_24h
        APP_CONFIG["user_preferences"]["date_format"] = prefs.date_format
        logger.info("System operational parameters updated safely over API.")
        return {"status": "success", "message": "Global preferences applied successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to flash targeted system configuration options: {str(e)}")

@app.get("/download_clip")
def download_clip(clip_relative_path: str):
    """Secure direct download handler guarding against path-traversal attacks."""
    base_storage_dir = Path(APP_CONFIG["storage"]["clip_dir"])
    target_file_path = (base_storage_dir / clip_relative_path).resolve()

    if not str(target_file_path).startswith(str(base_storage_dir.resolve())):
        raise HTTPException(status_code=403, detail="Access denied. Directory traversal blocked.")

    if not target_file_path.exists() or not target_file_path.is_file():
        raise HTTPException(status_code=404, detail="Requested dashcam clip footage not found.")

    return FileResponse(
        path=target_file_path,
        media_type="video/mp4",
        filename=target_file_path.name
    )

if __name__ == "__main__":
    uvicorn.run("src.main:app", host=APP_CONFIG["network"]["bind_address"], port=APP_CONFIG["network"]["port"], workers=1)