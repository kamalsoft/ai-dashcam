# src/sensors/gps_reader.py
import serial
import pynmea2
import threading
import logging

logger = logging.getLogger("NanovianDashcam")

class GPSReader:
    def __init__(self, port="/dev/ttyUSB0", baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Latest thread-safe metrics
        self.latitude = 0.0
        self.longitude = 0.0
        self.speed_mph = 0.0
        self.is_fixed = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        logger.info(f"GPS monitoring system deployed on port: {self.port}")

    def _read_loop(self):
        while self.running:
            try:
                # Open serial channel to physical GPS module
                with serial.Serial(self.port, baudrate=self.baudrate, timeout=1) as ser:
                    for line in ser:
                        if not self.running:
                            break
                        try:
                            # Convert bytes to string sentence
                            sentence = line.decode('ascii', errors='ignore').strip()
                            
                            # $GPRMC contains absolute time, coordinates, and true ground speed
                            if sentence.startswith('$GPRMC'):
                                msg = pynmea2.parse(sentence)
                                if msg.status == 'A':  # 'A' means active GPS lock signal
                                    with self.lock:
                                        self.latitude = float(msg.latitude)
                                        self.longitude = float(msg.longitude)
                                        # Convert knots speed attribute to MPH
                                        self.speed_mph = float(msg.spd_over_grnd or 0.0) * 1.15078
                                        self.is_fixed = True
                                else:
                                    with self.lock:
                                        self.is_fixed = False
                        except Exception:
                            continue
            except (serial.SerialException, FileNotFoundError):
                # Handle disconnects gracefully; sleep and try reconnecting
                with self.lock:
                    self.is_fixed = False
                threading.Event().wait(2.0)

    def get_location_data(self):
        with self.lock:
            return {
                "lat": self.latitude,
                "lon": self.longitude,
                "speed": self.speed_mph,
                "has_fix": self.is_fixed
            }

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)