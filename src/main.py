import os
import sys
import time
import logging
import threading
import cv2
import numpy as np
from fastapi import FastAPI, Response, status

# Set up clean system logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("DashcamServer")

app = FastAPI(title="LYNCUS Dashcam Engine")

# Global state sharing framework
class CameraState:
    def __init__(self):
        self.latest_frame = None
        self.lock = threading.Lock()
        self.is_running = False
        self.picam_instance = None
        self.v4l2_instance = None

state = CameraState()

def initialize_camera():
    """
    Attempts to initialize the native Raspberry Pi 5 Picamera2 stack.
    If it fails or is missing, drops back to a generic V4L2 OpenCV device pipeline.
    """
    # 1. Primary Path: Try Native Pi 5 Hardware Acceleration
    try:
        logger.info("Initializing native Raspberry Pi 5 camera layer (Picamera2)...")
        from picamera2 import Picamera2
        
        picam = Picamera2()
        # Use the exact video configuration matrix verified during diagnostics
        config = picam.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
        picam.configure(config)
        picam.start()
        
        state.picam_instance = picam
        logger.info("--- NATIVE PICAMERA2 PIPELINE ACTIVE ---")
        return "picamera2"
    
    except Exception as e:
        logger.warning(f"Picamera2 initialization bypassed or failed: {e}")
        if state.picam_instance:
            try: state.picam_instance.close()
            except: pass
            state.picam_instance = None

    # 2. Secondary Fallback Path: Generic V4L2 Device Loop (USB Webcams / Legacy)
    logger.info("Attempting fallback connection to /dev/video0 via OpenCV V4L2...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        state.v4l2_instance = cap
        logger.info("--- FALLBACK V4L2 CAPTURE ENGINE ACTIVE ---")
        return "v4l2"
    
    logger.error("All hardware streaming initialization strategies exhausted.")
    return None

def capture_worker():
    """
    Dedicated thread execution loop to keep frame capture timing detached 
    from FastAPI network performance overheads.
    """
    backend = initialize_camera()
    if not backend:
        logger.error("Capture thread exiting: No working camera device available.")
        return

    state.is_running = True
    logger.info("Frame acquisition loop running smoothly.")

    while state.is_running:
        try:
            if backend == "picamera2":
                # Grab native RGB888 numpy matrix from PiSP pipeline
                rgb_frame = state.picam_instance.capture_array()
                # Transcode back to standard OpenCV BGR space for uniformity
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            else:
                # Standard V4L2 capture
                ret, bgr_frame = state.v4l2_instance.read()
                if not ret:
                    time.sleep(0.01)
                    continue

            # Thread-safe storage assignment
            with state.lock:
                state.latest_frame = bgr_frame.copy()

        except Exception as e:
            logger.error(f"Error inside live acquisition thread loop: {e}")
            time.sleep(0.1)

        # Cap frame tracking load around ~30 FPS to conserve Pi 5 CPU overhead
        time.sleep(0.033)

    # Resource teardown sequence on app shutdown
    logger.info("Cleaning up active camera pipeline resources...")
    if state.picam_instance:
        try:
            state.picam_instance.stop()
            state.picam_instance.close()
        except: pass
    if state.v4l2_instance:
        state.v4l2_instance.release()

@app.on_event("startup")
def startup_event():
    """Launches frame tracking on its own decoupled thread thread layer."""
    t = threading.Thread(target=capture_worker, daemon=True)
    t.start()

@app.on_event("shutdown")
def shutdown_event():
    """Signals background execution loop to gracefully close physical devices."""
    state.is_running = False

@app.get("/video_frame.jpg")
def get_video_frame():
    """
    Serves the latest processed camera image frame directly as a JPEG binary.
    Falls back to a HTTP 503 response if the hardware layer is offline.
    """
    with state.lock:
        if state.latest_frame is None:
            return Response(
                content="Camera initialization pending or source unavailable.",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        # Deepcopy matrix to release read lock immediately
        frame_matrix = state.latest_frame.copy()

    # Encode internal memory matrix to binary stream distribution format
    success, encoded_jpeg = cv2.imencode(".jpg", frame_matrix)
    if not success:
        return Response(
            content="Matrix compression error.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    return Response(content=encoded_jpeg.tobytes(), media_type="image/jpeg")

@app.get("/list_incidents")
def list_incidents():
    """Mock metadata telemetry log integration endpoint."""
    return {"status": "active", "incidents": [], "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    # Force offscreen headless rendering platform profile
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    uvicorn.run(app, host="0.0.0.0", port=8000)