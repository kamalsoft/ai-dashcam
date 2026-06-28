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

class CameraState:
    def __init__(self):
        self.latest_frame = None       # Clean local image matrix (BGR space)
        self.annotated_frame = None    # Image matrix with YOLO bounding boxes
        self.lock = threading.Lock()
        self.is_running = False
        self.picam_instance = None
        self.v4l2_instance = None
        self.i2c_bus = None
        self.yolo_model = None

state = CameraState()

# Motorized Hardware Constants (I2C)
I2C_BUS_INDEX = 1        # Standard Raspberry Pi 5 I2C Hardware Pins (SDA/SCL)
MOTOR_I2C_ADDR = 0x40    # Targeted controller register address base (e.g., PCA9685)

def init_motor_hardware():
    """Maps system device nodes to expose active I2C channels."""
    try:
        state.i2c_bus = SMBus(I2C_BUS_INDEX)
        logger.info(f"Connected to physical I2C interface bus on index {I2C_BUS_INDEX}")
    except Exception as e:
        logger.warning(f"I2C Hardware Layer bypassed (Check connections or enable via raspi-config): {e}")

def command_motor_movement(pan_angle: int, tilt_angle: int):
    """
    Sends raw hexadecimal positional tracking steps to the driver base.
    """
    if state.i2c_bus is None:
        return
    try:
        # Placeholder for target register mapping logic
        logger.info(f"Motor positioning packet generated -> Pan: {pan_angle}°, Tilt: {tilt_angle}°")
    except Exception as e:
        logger.error(f"Failed to transmit I2C motor bus communication packet: {e}")

def initialize_camera():
    """
    Sets up Picamera2 using a true wide-angle widescreen resolution layout (16:9).
    """
    try:
        logger.info("Initializing native Raspberry Pi 5 wide-angle camera layer...")
        from picamera2 import Picamera2
        
        picam = Picamera2()
        
        # Configure widescreen resolution mapping (1280x720) to maintain true horizontal FOV
        config = picam.create_video_configuration(main={"format": "RGB888", "size": (1280, 720)})
        picam.configure(config)
        picam.start()
        
        state.picam_instance = picam
        logger.info("--- NATIVE WIDE-ANGLE PICAMERA2 PIPELINE ACTIVE ---")
        return "picamera2"
    
    except Exception as e:
        logger.warning(f"Picamera2 initialization sequence dropped: {e}")
        if state.picam_instance:
            try: state.picam_instance.close()
            except: pass
            state.picam_instance = None

    # Secondary Fallback Loop over V4L2 Device Nodes
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
    """Acquires image frames continuously from the physical hardware device."""
    backend = initialize_camera()
    if not backend:
        logger.error("Acquisition worker exiting: No video pipeline stream accessible.")
        return

    state.is_running = True
    logger.info("Frame acquisition worker thread running smoothly.")

    while state.is_running:
        try:
            if backend == "picamera2":
                # Extract wide-angle raw image sequence
                rgb_frame = state.picam_instance.capture_array()
                # Correct color spaces instantly by aligning to traditional BGR space
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            else:
                ret, bgr_frame = state.v4l2_instance.read()
                if not ret:
                    time.sleep(0.01)
                    continue

            with state.lock:
                state.latest_frame = bgr_frame.copy()

        except Exception as e:
            logger.error(f"Error executing raw pipeline matrix grab: {e}")
            time.sleep(0.1)

        time.sleep(0.01)

    # Teardown sequences
    logger.info("Releasing camera hardware allocations...")
    if state.picam_instance:
        try:
            state.picam_instance.stop()
            state.picam_instance.close()
        except: pass
    if state.v4l2_instance:
        state.v4l2_instance.release()

def ai_inference_worker():
    """
    Decoupled processing engine running YOLOv8 object detection tracking
    without dragging down the main capture framework frequency.
    """
    try:
        logger.info("Loading object detection model targets into memory...")
        from ultralytics import YOLO
        # Using yolov8n.pt (Nano version) to maximize frame compilation speeds on the Pi 5
        state.yolo_model = YOLO("yolov8n.pt")
        logger.info("--- DASHCAM AI INFERENCE ENGINE ONLINE ---")
    except Exception as e:
        logger.error(f"Failed to compile AI object models: {e}")
        return

    while state.is_running:
        if state.latest_frame is None:
            time.sleep(0.05)
            continue

        try:
            with state.lock:
                frame_to_process = state.latest_frame.copy()

            # Filter classes to extract typical road environments:
            # 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck, 9: traffic light
            results = state.yolo_model(frame_to_process, verbose=False, classes=[0, 2, 3, 5, 7, 9])
            
            # Generate the bounding box layout overlay matrix
            annotated = results[0].plot()

            with state.lock:
                state.annotated_frame = annotated

        except Exception as e:
            logger.error(f"Error inside AI vision tracking thread: {e}")
            time.sleep(0.1)

        time.sleep(0.01)

def generate_mjpeg_stream():
    """Yields continuous boundary frames to provide a real-time stream."""
    logger.info("Client connected to live dashboard AI stream channel.")
    while state.is_running:
        with state.lock:
            # Prioritize the AI-annotated stream frame, fallback to raw matrix if booting up
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
    # App Initialization Sequences
    init_motor_hardware()
    
    capture_thread = threading.Thread(target=capture_worker, daemon=True)
    capture_thread.start()
    
    ai_thread = threading.Thread(target=ai_inference_worker, daemon=True)
    ai_thread.start()
    
    yield
    # App Shutdown Sequences
    state.is_running = False
    if state.i2c_bus:
        try: state.i2c_bus.close()
        except: pass

app = FastAPI(title="LYNCUS Dashcam Engine", lifespan=lifespan_handler)

@app.get("/video_frame.jpg")
def get_video_frame():
    """Serves a static snapshot frame with the active layers applied."""
    with state.lock:
        frame_matrix = state.annotated_frame if state.annotated_frame is not None else state.latest_frame
        if frame_matrix is None:
            return Response(content="Camera pipeline starting...", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        img_copy = frame_matrix.copy()

    success, encoded_jpeg = cv2.imencode(".jpg", img_copy)
    if not success:
        return Response(content="Matrix compression error.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response(content=encoded_jpeg.tobytes(), media_type="image/jpeg")

@app.get("/video_feed")
def get_live_video_feed():
    """Exposes the continuous real-time AI-annotated MJPEG live video stream."""
    return StreamingResponse(generate_mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/list_incidents")
def list_incidents():
    return {"status": "active", "incidents": [], "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    uvicorn.run(app, host="0.0.0.0", port=8000)