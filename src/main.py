# src/main.py
import os
import sys
import time
import logging
import threading
import queue
import cv2
import asyncio
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse, Response
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

from src.camera.pi_camera import PiCamera
from src.processing.analytics import ThreatAnalytics
from src.storage.circular_buffer import CircularBuffer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger("NanovianDashcam")

APP_CONFIG = {
    "storage": {
        "clip_dir": str(Path(__file__).resolve().parents[1] / "clips"),
        "max_storage_mb": 2000,
        "clip_duration_seconds": 45,
        "incident_clip_duration_seconds": 120,
    },
    "analytics": {
        "min_confidence": 0.40,
        "video_source": "/dev/video0",
        "fps": 50.0,
    },
    "network": {
        "bind_address": "0.0.0.0",
        "port": 8000,
    },
    "gps": {
        "latitude": None,
        "longitude": None,
    },
    "user_preferences": {
        "timezone": "America/Chicago",
        "time_format_24h": False,
        "date_format": "%Y-%m-%d",
        "custom_location_label": "Naperville, IL",
    }
}

class UserPreferencesSchema(BaseModel):
    custom_location_label: str
    time_format_24h: bool
    date_format: str


class GPSCoordinatesSchema(BaseModel):
    latitude: float
    longitude: float

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
        clip_path = clip_root / f"clip_{self._ts()}.mkv"
        try:
            self.camera.start_recording(str(clip_path))
        except Exception as e:
            logger.error("Failed to start normal clip recording: %s", e)
            self.active_normal_clip = None
            return
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
        clip_path = incident_root / "incident_footage.mkv"

        if frame is not None:
            cv2.imwrite(str(snapshot_path), frame)

        try:
            self.camera.write_pre_buffer_to_incident(str(clip_path))
        except Exception as e:
            logger.error("Failed to start incident clip recording: %s", e)
            return
        self.storage_manager.register_active_file(str(clip_path))

        self.active_incident = {
            "root_dir": str(incident_root),
            "clip_path": str(clip_path),
            "started_at": time.monotonic()
        }

    def _adas_worker_loop(self):
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
                self.storage_manager.enforce_retention_policy_async(".mkv")

                if self.active_incident:
                    elapsed = time.monotonic() - self.active_incident["started_at"]
                    if elapsed >= self.config["storage"]["incident_clip_duration_seconds"]:
                        logger.info("Incident window expired. Resuming standard loop operations.")
                        self.camera.stop_recording()
                        self.storage_manager.unregister_active_file(self.active_incident["clip_path"])
                        self.active_incident = None
                        self._start_normal_clip()
                else:
                    if self.active_normal_clip and (time.monotonic() - self.normal_clip_started_at) >= self.config["storage"]["clip_duration_seconds"]:
                        self.camera.stop_recording()
                        self.storage_manager.unregister_active_file(self.active_normal_clip)
                        self.active_normal_clip = None
                        self._start_normal_clip()

                ok = self.camera.update_frame()
                if not ok: 
                    time.sleep(0.05)
                    continue

                raw_frame = self.camera.get_latest_frame()
                if raw_frame is not None:
                    try:
                        self.ai_frame_queue.put_nowait(raw_frame)
                    except queue.Full:
                        pass

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

@app.get("/", response_class=HTMLResponse)
def serve_dashboard_home_screen():
    template_path = Path(__file__).parent / "templates" / "index.html"
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Web interface HTML template asset missing.")
    return template_path.read_text()

@app.get("/video_feed")
async def video_feed():
    async def frame_generator():
        while orchestrator.is_running:
            if orchestrator.latest_encoded_frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + orchestrator.latest_encoded_frame + b'\r\n')
            await asyncio.sleep(0.05)
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
    )


@app.get("/video_frame.jpg")
def video_frame_snapshot():
    frame = orchestrator.latest_encoded_frame
    if not frame:
        raise HTTPException(status_code=503, detail="Live frame not available yet.")
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

@app.get("/list_incidents")
def list_incidents():
    incidents_dir = Path(APP_CONFIG["storage"]["clip_dir"]) / "incidents"
    if not incidents_dir.exists():
        return {"incidents": []}
    
    entries = []
    for p in sorted(incidents_dir.iterdir(), reverse=True):
        if p.is_dir():
            has_snap = (p / "snapshot.jpg").exists()
            entries.append({
                "incident_id": p.name,
                "has_snapshot": has_snap,
                "video_url": f"/api/media/stream?type=incidents&id={p.name}",
                "snapshot_url": f"/api/media/snapshot?id={p.name}" if has_snap else None
            })
    return {"incidents": entries}

# --- ADVANCED HTTP BYTE-RANGE VIDEO STREAMING ROUTER ---
@app.get("/api/media/stream")
def stream_dashcam_video(type: str, id: str, range: str = Header(None)):
    """Natively handles partial byte-range tracking requests for clean in-browser streaming."""
    base_dir = Path(APP_CONFIG["storage"]["clip_dir"]) / type / id
    video_path = base_dir / "incident_footage.mkv"
    
    if type == "normal":
        base_dir = Path(APP_CONFIG["storage"]["clip_dir"]) / "normal"
        video_path = base_dir / id

    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(status_code=404, detail="Video track entry target missing.")

    file_size = video_path.stat().st_size
    start, end = 0, file_size - 1

    if range:
        parts = range.replace("bytes=", "").split("-")
        if parts[0]: start = int(parts[0])
        if parts[1]: end = int(parts[1])

    chunk_size = (end - start) + 1
    
    def video_chunk_generator():
        with open(video_path, "rb") as video_file:
            video_file.seek(start)
            bytes_left = chunk_size
            while bytes_left > 0:
                chunk = video_file.read(min(128 * 1024, bytes_left))
                if not chunk:
                    break
                bytes_left -= len(chunk)
                yield chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
        "Content-Type": "video/webm"  # Direct injection mapping forces unified playback behavior
    }
    return StreamingResponse(video_chunk_generator(), status_code=206, headers=headers)

@app.get("/api/media/snapshot")
def get_incident_snapshot(id: str):
    snap_path = Path(APP_CONFIG["storage"]["clip_dir"]) / "incidents" / id / "snapshot.jpg"
    if not snap_path.exists():
        raise HTTPException(status_code=404, detail="Snapshot file not found.")
    return FileResponse(snap_path, media_type="image/jpeg")

@app.post("/update_preferences")
def update_preferences(prefs: UserPreferencesSchema):
    try:
        APP_CONFIG["user_preferences"]["custom_location_label"] = prefs.custom_location_label
        APP_CONFIG["user_preferences"]["time_format_24h"] = prefs.time_format_24h
        APP_CONFIG["user_preferences"]["date_format"] = prefs.date_format
        logger.info("System operational parameters updated safely over API.")
        return {"status": "success", "message": "Global preferences applied successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to flash configuration: {str(e)}")


@app.post("/update_gps")
def update_gps_coordinates(gps: GPSCoordinatesSchema):
    try:
        APP_CONFIG["gps"]["latitude"] = float(gps.latitude)
        APP_CONFIG["gps"]["longitude"] = float(gps.longitude)
        logger.info(
            "GPS coordinates updated: latitude=%s longitude=%s",
            APP_CONFIG["gps"]["latitude"],
            APP_CONFIG["gps"]["longitude"],
        )
        return {
            "status": "success",
            "message": "GPS coordinates applied to live HUD and recordings.",
            "gps": APP_CONFIG["gps"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply GPS coordinates: {str(e)}")

@app.post("/api/system/stop")
def stop_recording_lifecycle():
    with orchestrator.camera._lock:
        if not orchestrator.active_normal_clip and not orchestrator.active_incident:
            return {"status": "ignored", "message": "Recording loops are already dormant."}
        
        logger.warning("Administrative Halt requested over API. Dropping file records...")
        orchestrator.camera.stop_recording()
        
        if orchestrator.active_normal_clip:
            orchestrator.storage_manager.unregister_active_file(orchestrator.active_normal_clip)
            orchestrator.active_normal_clip = None
        orchestrator.active_incident = None
        return {"status": "success", "message": "Dashcam pipeline recording safely stopped."}

@app.post("/api/system/start")
def start_recording_lifecycle():
    with orchestrator.camera._lock:
        if orchestrator.active_normal_clip or orchestrator.active_incident:
            raise HTTPException(status_code=400, detail="Recording pipeline is already active.")
        
        logger.info("Administrative Start requested over API. Spinning up channels...")
        orchestrator._start_normal_clip()
        return {"status": "success", "message": "Continuous loops re-engaged successfully."}

@app.post("/api/system/restart")
def restart_recording_lifecycle():
    with orchestrator.camera._lock:
        logger.info("Cycle reboot requested over API. Flushing IO streams...")
        if orchestrator.active_normal_clip or orchestrator.active_incident:
            orchestrator.camera.stop_recording()
            if orchestrator.active_normal_clip:
                orchestrator.storage_manager.unregister_active_file(orchestrator.active_normal_clip)
                orchestrator.active_normal_clip = None
            orchestrator.active_incident = None
            
        orchestrator._start_normal_clip()
        return {"status": "success", "message": "Pipeline capture cycled and rebooted successfully."}

@app.post("/api/storage/clear")
def clear_all_saved_media():
    with orchestrator.camera._lock:
        logger.warning("CRITICAL: Administrative Storage Clear commanded over API portal.")
        orchestrator.camera.stop_recording()
        
        base_dir = Path(APP_CONFIG["storage"]["clip_dir"])
        for sub_folder in ["normal", "incidents"]:
            target_path = base_dir / sub_folder
            if target_path.exists():
                shutil.rmtree(target_path)
            target_path.mkdir(parents=True, exist_ok=True)
            
        orchestrator.active_normal_clip = None
        orchestrator.active_incident = None
        orchestrator.storage_manager.tracked_files = []
        
        orchestrator._start_normal_clip()
        return {"status": "success", "message": "All saved loops and locked incidents dropped. Storage clear."}

@app.post("/api/storage/clean")
def clean_stale_storage():
    logger.info("On-demand maintenance sweep initialized manually over API gateway.")
    orchestrator.storage_manager.enforce_retention_policy_async(".mkv")
    return {"status": "success", "message": "Garbage collector sweep finished processing storage array profiles."}

@app.get("/download_clip")
def download_clip(type: str, id: str):
    """Secure direct download asset handler routing for clean binary file delivery."""
    base_dir = Path(APP_CONFIG["storage"]["clip_dir"]) / type / id
    video_path = base_dir / "incident_footage.mkv"
    
    if type == "normal":
        video_path = Path(APP_CONFIG["storage"]["clip_dir"]) / "normal" / id

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Requested file asset missing.")

    return FileResponse(path=video_path, media_type="video/x-matroska", filename=video_path.name)

if __name__ == "__main__":
    uvicorn.run("src.main:app", host=APP_CONFIG["network"]["bind_address"], port=APP_CONFIG["network"]["port"], workers=1)