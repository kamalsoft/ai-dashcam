import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CamTest")

try:
    from picamera2 import Picamera2
    logger.info("Initializing Picamera2 with standard video configuration...")
    picam = Picamera2()
    
    # Use create_video_configuration which supports array-friendly formats natively
    config = picam.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam.configure(config)
    
    picam.start()
    logger.info("SUCCESS: Camera started! Pulling a live frame matrix...")
    
    # Capture directly as an array
    frame = picam.capture_array()
    logger.info("Frame matrix captured successfully! Shape: %s", str(frame.shape))
    
    picam.stop()
    logger.info("--- HARDWARE PIPELINE PERFECT ---")
except Exception as e:
    logger.error("PIPELINE FAILED AT RUNTIME!", exc_info=True)