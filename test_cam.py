import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CamTest")

logger.info("Testing Python version: %s", sys.version)

try:
    logger.info("Attempting to import picamera2...")
    from picamera2 import Picamera2
    logger.info("SUCCESS: picamera2 imported perfectly!")
    
    logger.info("Attempting to initialize the IMX219 hardware layer...")
    picam = Picamera2()
    picam.configure(picam.create_preview_configuration(main={"format": "BGR24", "size": (640, 480)}))
    picam.start()
    
    logger.info("SUCCESS: Camera started! Capturing 1 diagnostic frame...")
    frame = picam.capture_array()
    logger.info("Frame matrix captured successfully. Shape: %s", str(frame.shape))
    
    picam.stop()
    logger.info("--- HARDWARE PIPELINE PERFECT ---")
except Exception as e:
    logger.error("PIPELINE FAILED AT RUNTIME!", exc_info=True)