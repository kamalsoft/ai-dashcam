# src/main.pyquirements
import os
import sysed Targets
import multiprocessingtop
- Raspberry Pi-style device with camera support
# 1. CRITICAL CONTEXT SETUP: Enforce macOS process isolation & thread limits
# This MUST happen before any third-party frameworks (like PyTorch or Ultralytics) are imported.
if sys.platform == "darwin":
    try:um Requirements
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        passe camera input or video source
- Local storage for clips and snapshots
os.environ["YOLO_VERBOSE"] = "False"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
- 8 GB RAM or more
import timege for incident media
import loggingmera sensor or stable webcam
import signalle of sustaining real-time frame processing
import sys
import traceback
sources:
def _incident_timestamp() -> str:webcam
    return time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"am
rce
# 2. DEFERRED IMPORTS: Loaded safely after process start methods and environment bounds are set- Pi camera-compatible input
from src.camera.mac_camera import MacCamera
from src.processing.analytics import ThreatAnalytics
from src.storage.circular_buffer import CircularBufferRecommended:

# Configure structured system logging retention
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'## Performance Considerations
)d
logger = logging.getLogger("DashcamCore")es storage and processing demand
mance depends on CPU/GPU availability
# Configuration Block
APP_CONFIG = {
    "storage": {
        "clip_dir": "./mock_dashcam_clips",- Use a stable tripod or fixed mount for accurate video capture
        "max_storage_mb": 1000,e consistent when possible
        "clip_duration_seconds": 45  # Keeps file sizes cleanly over 2MB thresholdy Pi deployments, ensure adequate cooling    },    "analytics": {        "min_confidence": 0.45,        "video_source": "./assets/test_dashcam.mp4"    },    "network": {        "bind_address": "0.0.0.0",        "port": 8080    },}class DashcamOrchestrator:    def __init__(self, config: dict):        self.config = config        self.is_running = True        self.shutdown_event = threading.Event()        self.prebuffer = deque()        self.active_incident = None        self.target_fps = float(config.get("fps", 20.0))        self.frame_period = 1.0 / max(self.target_fps, 1.0)        self.prebuffer_seconds = float(config["storage"].get("incident_prebuffer_seconds", 12.0))        self.postbuffer_seconds = float(config["storage"].get("incident_postbuffer_seconds", 2.0))        self.max_prebuffer_frames = int(self.target_fps * self.prebuffer_seconds)    def _incident_timestamp(self) -> str:        return time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"    def _create_incident_dir(self) -> str:        base_dir = Path(self.config["storage"]["clip_dir"]) / "incidents"        base_dir.mkdir(parents=True, exist_ok=True)        stamp = self._incident_timestamp()        incident_dir = base_dir / f"incident_{stamp}"        incident_dir.mkdir(parents=True, exist_ok=True)        return str(incident_dir), stamp    def _append_prebuffer_frame(self, frame):        self.prebuffer.append(frame.copy())        while len(self.prebuffer) > self.max_prebuffer_frames:            self.prebuffer.popleft()    def _start_incident(self, frame, telemetry):        incident_dir, stamp = self._create_incident_dir()        snapshot_path = os.path.join(incident_dir, f"snapshot_{stamp}.jpg")        clip_path = os.path.join(incident_dir, f"clip_{stamp}.avi")        self.camera.save_incident_snapshot(snapshot_path, frame, telemetry)        self.camera.start_recording(clip_path, fps=self.target_fps)        for buffered_frame in self.prebuffer:            self.camera.writer.write(buffered_frame)        self.active_incident = {            "dir": incident_dir,            "stamp": stamp,            "clip_path": clip_path,            "last_seen": time.monotonic(),        }    def _exact_sleep_until(self, deadline: float):        now = time.monotonic()        remaining = deadline - now        if remaining > 0:            time.sleep(remaining)    def start_wifi_portal(self):        """Spins up an internal media server accessible over the vehicle's hotspot."""        target_dir = os.path.abspath(self.config["storage"]["clip_dir"])        os.makedirs(target_dir, exist_ok=True)                class IncidentMediaHandler(http.server.SimpleHTTPRequestHandler):            def __init__(self, *args, **kwargs):                super().__init__(*args, directory=target_dir, **kwargs)        def server_worker():            socketserver.TCPServer.allow_reuse_address = True            try:                with socketserver.TCPServer((self.config["network"]["bind_address"], self.config["network"]["port"]), IncidentMediaHandler) as httpd:                    logger.info(f"[WiFi Portal] Server Live at http://192.168.4.1:{self.config['network']['port']}/")                    httpd.serve_forever()            except Exception as e:                logger.error(f"[WiFi Portal] Server thread encountered an issue: {e}")        portal_thread = threading.Thread(target=server_worker, daemon=True)        portal_thread.start()    def run_lifecycle(self):        self.start_wifi_portal()        self.camera.initialize(self.config["analytics"])        next_deadline = time.monotonic()        try:            while self.is_running and not self.shutdown_event.is_set():                loop_start = time.monotonic()                ok = self.camera.update_frame()                if not ok:                    self._request_shutdown("camera failure")                    break                frame = self.camera.latest_annotated_frame                    self._append_prebuffer_frame(frame)                threat = self.analytics.process_inference_metadata(self.camera.get_ai_metadata())                if threat and self.active_incident is None and frame is not None:                    self._start_incident(frame, telemetry)                if self.active_incident and self.camera.writer and frame is not None:                    self.camera.writer.write(frame)                    self.active_incident["last_seen"] = time.monotonic()                if self.active_incident and (time.monotonic() - self.active_incident["last_seen"]) >= self.postbuffer_seconds:                    self.camera.stop_recording()                    self.active_incident = None                next_deadline += self.frame_period                self._exact_sleep_until(next_deadline)                if time.monotonic() - loop_start > 2.0 * self.frame_period:                    next_deadline = time.monotonic()        finally:            self._shutdown()    def _shutdown(self):        try:            if self.active_incident:                self.camera.stop_recording()        finally:            self.camera.close()            if hasattr(self, "portal_server") and self.portal_server:                self.portal_server.shutdown()                self.portal_server.server_close()def install_crash_guards(orchestrator):    def handle_signal(signum, _frame):        logger.warning(f"Shutdown signal received: {signum}")
        orchestrator.shutdown_event.set()
        orchestrator.is_running = False

    def handle_exception(exc_type, exc, tb):
        logger.error("Unhandled exception detected")
        logger.error("".join(traceback.format_exception(exc_type, exc, tb)))
        orchestrator.shutdown_event.set()
        orchestrator.is_running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    sys.excepthook = handle_exception

    def thread_excepthook(args):
        logger.error(f"Thread failure in {args.thread.name}: {args.exc_value}")
        orchestrator.shutdown_event.set()
        orchestrator.is_running = False

    threading.excepthook = thread_excepthook