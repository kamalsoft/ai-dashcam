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

# Reference your modular local hardware abstraction definitions
from src.camera.pi_camera import PiCamera

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("DashcamServer")

class ServerState:
    def __init__(self):
        self.camera_driver = None
        self.ai_annotated_frame = None
        self.lock = threading.Lock()
        self.is_running = False
        self.i2c_bus = None
        self.yolo_model = None

server_state = ServerState()

# Motorized Pan-Tilt Hardware Configurations (I2C interface)
I2C_BUS_INDEX = 1
MOTOR_I2C_ADDR = 0x40

def init_motor_bus():
    """Binds operational I2C buses to target pin nodes."""
    try:
        server_state.i2c_bus = SMBus(I2C_BUS_INDEX)
        logger.info(f"Exposed structural I2C communication lane over interface index {I2C_BUS_INDEX}")
    except Exception as e:
        logger.warning(f"I2C Bus registration bypassed (Hardware unattached or node closed): {e}")

def command_motor_movement(pan_angle: int, tilt_angle: int):
    """Fires coordinate step directives over the physical I2C matrix lines."""
    if server_state.i2c_bus is None:
        return
    try:
        logger.info(f"Motorized mount adjustment sent -> Pan: {pan_angle}°, Tilt: {tilt_angle}°")
        # server_state.i2c_bus.write_byte_data(MOTOR_I2C_ADDR, register, value)
    except Exception as e:
        logger.error(f"Failed to submit hardware command over I2C lines: {e}")

def frames_acquisition_loop():
    """Continuous thread task loops to update frames from the underlying PiCamera driver driver."""
    logger.info("Spawning decoupled hardware acquisition thread...")
    server_state.is_running = True
    
    while server_state.is_running:
        try:
            success = server_state.camera_driver.update_frame()
            if not success:
                time.sleep(0.01)
                continue
        except Exception as e:
            logger.error(f"Error updating matrix sequence from camera interface layer: {e}")
            time.sleep(0.1)
        time.sleep(0.01)

def ai_inference_loop():
    """Processes newly fetched frame matrices using YOLOv8 models."""
    try:
        logger.info("Assembling YOLOv8 Neural Target Matrix structures into RAM...")
        from ultralytics import YOLO
        server_state.yolo_model = YOLO("yolov8n.pt")
        logger.info("--- NATIVE DASHCAM AI VISION ENGINES ACTIVE ---")
    except Exception as e:
        logger.error(f"Aborted object detection pipeline startup: {e}")
        return

    while server_state.is_running:
        # Pull baseline frames directly out of your camera module abstraction class instance
        raw_frame = server_state.camera_driver.get_latest_frame()
        if raw_frame is None:
            time.sleep(0.05)
            continue

        try:
            # Classes targeted: 0 (person), 2 (car), 3 (motorcycle), 5 (bus), 7 (truck), 9 (traffic light)
            results = server_state.yolo_model(raw_frame, verbose=False, classes=[0, 2, 3, 5, 7, 9])
            processed_matrix = results[0].plot()

            # Optional Motor Tracking Demo Hooks:
            # If target vectors cross specific thresholds, fire I2C step adjustments
            # command_motor_movement(pan_angle=120, tilt_angle=30)

            with server_state.lock:
                server_state.ai_annotated_frame = processed_matrix
        except Exception as e:
            logger.error(f"Exception encountered inside edge AI execution sequence: {e}")
            time.sleep(0.1)
        time.sleep(0.01)

def generate_mjpeg_broadcast():
    """Generates continuous multipart stream data payloads for local networks."""
    logger.info("Client registered to receive live multipart AI video broadcast payload.")
    while server_state.is_running:
        with server_state.lock:
            frame = server_state.ai_annotated_frame if server_state.ai_annotated_frame is not None else server_state.camera_driver.get_latest_frame()

        if frame is None:
            time.sleep(0.033)
            continue

        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(buffer)).encode() + b'\r\n\r\n' +
               buffer.tobytes() + b'\r\n')
        time.sleep(0.033)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Configuration Sequence
    init_motor_bus()
    
    # Initialize your updated abstraction class module
    server_state.camera_driver = PiCamera()
    
    # Build a runtime configuration dictionary for your camera initialization layer
    runtime_config = {
        "analytics": {
            "frame_width": 1280,
            "frame_height": 720,
            "fps": 20.0,
            "video_source": "/dev/video0",
            "use_picamera2": True
        },
        "user_preferences": {
            "date_format": "%Y-%m-%d",
            "time_format_24h": True,
            "custom_location_label": "LYNCUS AI DASHCAM"
        }
    }
    
    server_state.camera_driver.initialize(runtime_config)
    
    # Launch tracking routines along decoupled worker execution levels
    threading.Thread(target=frames_acquisition_loop, daemon=True).start()
    threading.Thread(target=ai_inference_loop, daemon=True).start()
    
    yield
    # Shutdown Teardown Sequence
    server_state.is_running = False
    if server_state.camera_driver:
        server_state.camera_driver.close()
    if server_state.i2c_bus:
        try: server_state.i2c_bus.close()
        except: pass

app = FastAPI(title="LYNCUS Dashcam Engine Core", lifespan=lifespan)

@app.get("/video_feed")
def live_video_feed_route():
    """Exposes real-time high-resolution AI tracking streams over local network bounds."""
    return StreamingResponse(generate_mjpeg_broadcast(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_frame.jpg")
def snapshot_frame_route():
    frame = server_state.ai_annotated_frame if server_state.ai_annotated_frame is not None else server_state.camera_driver.get_latest_frame()
    if frame is None:
        return Response(content="Camera arrays initializing...", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    success, buffer = cv2.imencode(".jpg", frame)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    uvicorn.run(app, host="0.0.0.0", port=8000)