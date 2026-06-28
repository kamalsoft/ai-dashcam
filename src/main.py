import os
import sys
import time
import logging
import threading
import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, status
from fastapi.responses import StreamingResponse
from smbus2 import SMBus

# Set up clean system logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("DashcamServer")

# Global state sharing framework
class CameraState:
    def __init__(self):
        self.latest_frame = None       # Clear raw matrix
        self.annotated_frame = None    # Frame with AI bounding boxes
        self.lock = threading.Lock()
        self.is_running = False
        self.picam_instance = None
        self.v4l2_instance = None
        self.i2c_bus = None
        self.yolo_model = None

state = CameraState()

# Motorized Hardware Control Constants (I2C)
I2C_BUS_INDEX = 1        # Standard Pi 5 I2C interface pins (SDA/SCL)
MOTOR_I2C_ADDR = 0x40    # Default target address for PCA9685/Servo drivers

def init_motor_hardware():
    """Initializes the I2C bus channel for motorized pan-tilt logic."""
    try:
        state.i2c_bus = SMBus(I2C_BUS_INDEX)
        logger.info(f"Connected to I2C interface bus on index {I2C_BUS_INDEX}")
    except Exception as e:
        logger.warning(f"I2C Bus initialization bypassed (Hardware missing/unplugged): {e}")

def command_motor_movement(pan_angle: int, tilt_angle: int):
    """
    Sends raw programmatic positioning bits to the motorized driver base.
    """
    if state.i2c_bus is None:
        return
    try:
        # Example register mapping instructions to a PCA9685 controller
        # In a custom driver deployment, replace these with your specific registry maps
        logger.info(f"Motor command fired -> Pan: {pan_angle}°, Tilt: {tilt_angle}°")
        # state.i2c_bus.write_byte_data(MOTOR_I2C_ADDR, register, value)
    except Exception as e:
        logger.error(f"Failed to transmit I2C motor step data packet: {e}")

def initialize_camera():
    """
    Initializes Picamera2 using a wide-angle widescreen format profile.
    """
    try:
        logger.info("Initializing native Raspberry Pi 5 wide-angle camera layer...")
        from picamera2 import Picamera2
        
        picam = Picamera2()
        # Switch to 16:9 widescreen format (1280x720) to capture the wide lens FOV
        config = picam.create_video_configuration(main={"format": "RGB888", "size": (1280, 720)})
        picam.configure(config)
        picam.start()
        
        state.picam_instance = picam
        logger.info("--- NATIVE WIDE-ANGLE PICAMERA2 PIPELINE ACTIVE ---")
        return "picamera2"
    
    except Exception as e:
        logger.warning(f"Picamera2 wide-angle initialization failed: {e}")
        if state.picam_instance:
            try: state.picam_instance.close()
            except: pass
            state.picam_instance = None

    # Secondary Fallback
    logger.info("Attempting fallback connection to /dev/video0 via OpenCV V4L2...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        state.v4l2_instance = cap
        logger.info("--- FALLBACK V4L2 CAPTURE ENGINE ACTIVE ---")
        return "v4l2"
    
    return None

def capture_worker():
    """Dedicated thread execution loop to acquire frames from the hardware device."""
    backend = initialize_camera()
    if not backend:
        logger.error("Capture thread exiting: No working camera device available.")
        return

    state.is_running = True
    logger.info("Wide-angle frame acquisition loop running smoothly.")

    while state.is_running:
        try:
            if backend == "picamera2":
                # Raw layout array grab
                rgb_frame = state.picam_instance.capture_array()
                
                # FIX: Slice matrix directly to correct BGR format natively
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            else:
                ret, bgr_frame = state.v4l2_instance.read()
                if not ret:
                    time.sleep(0.01)
                    continue

            with state.lock:
                state.latest_frame = bgr_frame.copy()

        except Exception as e:
            logger.error(f"Error inside frame acquisition thread loop: {e}")
            time.sleep(0.1)

        time.sleep(0.01) # Maximize throughput

    # Cleanup
    if state.picam_instance:
        try:
            state.picam_instance.stop()
            state.picam_instance.close()
        except: pass
    if state.v4l2_instance:
        state.v4l2_instance.release()

def ai_inference_worker():
    """
    Dedicated core processing thread. Avoids blocking the camera 
    capture thread by pulling the latest frame and processing it with YOLOv8.
    """
    try:
        logger.info("Loading object detection model models into memory...")
        from ultralytics import YOLO
        # Using yolov8n.pt (Nano) to maximize frame processing speeds on the Pi 5 CPU/GPU matrix
        state.yolo_model = YOLO("yolov8n.pt")
        logger.info("--- AI YOLOV8 CORE SYSTEM ONLINE ---")
    except Exception as e:
        logger.error(f"Failed to spin up AI Inference subsystem framework: {e}")
        return

    while state.is_running:
        if state.latest_frame is None:
            time.sleep(0.05)
            continue

        try:
            # Thread-safe snapshot copy
            with state.lock:
                frame_to_process = state.latest_frame.copy()

            # Execute real-time lightweight local target prediction scanning
            # We filter predictions to classes: 0 (person), 2 (car), 3 (motorcycle), 5 (bus), 7 (truck), 9 (traffic light)
            results = state.yolo_model(frame_to_process, verbose=False, classes=[0, 2, 3, 5, 7, 9])
            
            # Extract an annotated frame with bounding boxes plotted
            annotated = results[0].plot()

            # Optional Motor Tracking Demo: If an object gets dangerously close to a screen edge,
            # we can trigger I2C movement commands to align the motorized mount.
            # Example: command_motor_movement(pan_angle=90, tilt_angle=45)

            with state.lock:
                state.annotated_frame = annotated

        except Exception as e:
            logger.error(f"Error executing frame AI prediction loop: {e}")
            time.sleep(0.1)

        time.sleep(0.01)

def generate_mjpeg_stream():
    """Continuous loop generating the live MJPEG video broadcast."""
    logger.info("Client attached to live dashboard AI stream feed channel.")
    while state.is_running:
        with state.lock:
            # Fall back to raw capture frames if the AI tracking pipeline is still booting up
            frame_matrix = state.annotated_frame if state.annotated_frame is not None else state.latest_frame
        
        if frame_matrix is None:
            time.sleep(0.033)
            continue

        success, encoded_jpeg = cv2.imencode(".jpg", frame_matrix)
        if not success:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(encoded_jpeg)).encode() + b'\r\n\r\n' +
               encoded_jpeg.tobytes() + b'\r\n')
        
        time.sleep(0.033)

@asynccontextmanager
async def lifespan_handler(app: FastAPI):
    # Startup Setup Procedures
    init_motor_hardware()
    
    capture_thread = threading.Thread(target=capture_worker, daemon=True)
    capture_thread.start()
    
    ai_thread = threading.Thread(target=ai_inference_worker, daemon=True)
    ai_thread.start()
    
    yield
    # Shutdown Processing Sequences
    state.is_running = False
    if state.i2c_bus:
        try: state.i2c_bus.close()
        except: pass

app = FastAPI(title="LYNCUS Dashcam Engine", lifespan=lifespan_handler)

@app.get("/video_frame.jpg")
def get_video_frame():
    """Serves a single snapshot frame."""
    with state.lock:
        frame_matrix = state.annotated_frame if state.annotated_frame is not None else state.latest_frame
        if frame_matrix is None:
            return Response(content="Camera initialization pending.", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        img_copy = frame_matrix.copy()

    success, encoded_jpeg = cv2.imencode(".jpg", img_copy)
    if not success:
        return Response(content="Compression error.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response(content=encoded_jpeg.tobytes(), media_type="image/jpeg")

@app.get("/video_feed")
def get_live_video_feed():
    """Exposes the real-time AI-annotated MJPEG live video broadcast pipeline."""
    return StreamingResponse(generate_mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/list_incidents")
def list_incidents():
    return {"status": "active", "incidents": [], "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    uvicorn.run(app, host="0.0.0.0", port=8000)