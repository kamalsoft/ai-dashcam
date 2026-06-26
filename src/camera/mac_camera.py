# src/camera/mac_camera.py
import time
import os
import cv2
import numpy as np
import reverse_geocoder as rg
from ultralytics import YOLO
from src.camera.base_camera import BaseCamera
from collections import deque

class MacCamera(BaseCamera):
    def __init__(self):
        self.cap = None
        self.writer = None
        self.model = None
        self.config = {}
        self.current_metadata = []
        self.latest_annotated_frame = None
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30.0
        self.target_fps = 20.0
        self.pre_buffer = deque(maxlen=300)

        # --- ADAS Tracker Constants ---
        self.FOCAL_LENGTH_PIXELS = 500.0
        self.CLASS_MAPPINGS = {
            0: "Pedestrian", 2: "Car", 3: "Motorcycle", 
            5: "Bus", 7: "Truck", 9: "Traffic Light", 11: "Stop Sign"
        }
        self.REAL_HEIGHTS = {
            "Pedestrian": 1.75, "Car": 1.5, "Motorcycle": 1.4,
            "Bus": 3.2, "Truck": 3.0, "Traffic Light": 1.0, "Stop Sign": 0.75
        }

        # --- Kinematic & Optical Flow Registers ---
        self.prev_gray = None
        self.current_speed_kmh = 0.0
        self.current_heading = "N"
        self.current_address = "Naperville, IL"
        
        # Simulated GPS coordinate starting baseline
        self.prev_lat = 41.77940
        self.prev_lon = -88.15680
        self.prev_time = time.time()

        self._last_geocode_lookup = 0.0
        self._geocode_interval_sec = 1.0

    def initialize(self, config: dict) -> None:
        print("[Local-First Engine] Initializing vehicular tracking matrix...")
        self.config = config

        video_asset = self.config.get("video_source", "./assets/test_dashcam.mp4")
        use_webcam = True

        if os.path.exists(video_asset):
            print(f"[Video Engine] Loading local driving asset pipeline: {video_asset}")
            self.cap = cv2.VideoCapture(video_asset)
            if self.cap.isOpened():
                self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

                source_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
                if source_fps > 0 and np.isfinite(source_fps):
                    self.target_fps = source_fps
                else:
                    self.target_fps = float(config.get("fps", 20.0))
                use_webcam = False
            else:
                print(f"[Video Engine] OpenCV failed to parse {video_asset} (corrupted or unreadable format).")
                self.cap.release()

        if use_webcam:
            print("[Video Engine] Deploying safe fallback. Defaulting to MacBook webcam live feed.")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.fps = float(self.config.get("fps", 30.0))

        self.fps = float(self.config.get("fps", self.fps))

        if not self.cap.isOpened():
            raise RuntimeError("Video capture layer failed completely (both asset file and hardware camera un-initializable).")

        LOCAL_MODEL_PATH = os.path.abspath("./models/yolov8n.pt")
        self.model = YOLO(LOCAL_MODEL_PATH)
        print("[Local-First Engine] Offline Speed, Heading, & Optical Flow Translators Online.")

    def _get_offline_address(self, lat: float, lon: float) -> str:
        try:
            # Force single-process mode to avoid multiprocessing join interruption
            results = rg.search((lat, lon), mode=1, verbose=False)
            if results:
                r = results[0]
                return f"{r.get('name', '')}, {r.get('admin1', '')}"
        except Exception:
            return self.current_address
        return self.current_address

    def _calculate_optical_flow_speed(self, current_frame: np.ndarray) -> float:
        """Uses Farneback Dense Optical Flow to estimate speed from road texture motion."""
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Isolate the lower 35% section of the screen where the asphalt sweeps down
        h, w = gray.shape
        road_roi = gray[int(h * 0.65):int(h * 0.95), :]
        
        if self.prev_gray is None or self.prev_gray.shape != road_roi.shape:
            self.prev_gray = road_roi
            return self.current_speed_kmh

        # Compute vertical pixel vectors
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, road_roi, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        flow_y = flow[..., 1]
        significant_motion = flow_y[flow_y > 0.5]
        
        if len(significant_motion) > 100:
            avg_pixel_velocity = np.mean(significant_motion)
            CALIBRATION_FACTOR = 12.5 
            raw_speed = avg_pixel_velocity * CALIBRATION_FACTOR
        else:
            raw_speed = 0.0

        # Run an exponential smoothing filter over velocity readings to keep HUD stable
        self.current_speed_kmh = round((0.85 * self.current_speed_kmh) + (0.15 * raw_speed), 1)
        self.prev_gray = road_roi
        return self.current_speed_kmh

    def _apply_hud(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        h, w = frame.shape[:2]
        band_h = 72
        cv2.rectangle(overlay, (0, 0), (w, band_h), (0, 0, 0), -1)
        blended = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        speed = getattr(self, "current_speed_kmh", 0.0)
        heading = getattr(self, "current_heading", "N")
        addr = getattr(self, "current_address", "Unknown")

        cv2.putText(blended, f"{ts}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(blended, f"Speed: {speed:.1f} km/h  Heading: {heading}", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(blended, f"{addr}", (420, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 120), 1, cv2.LINE_AA)
        return blended

    def update_frame(self) -> bool:
        if not self.cap or not self.cap.isOpened():
            return False
            
        ret, frame = self.cap.read()
        if not ret:
            # Loop the local video asset infinitely if it hits EOF during development testing
            if self.config.get("video_source") and os.path.exists(self.config.get("video_source")):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret: return False
            else:
                return False

        self.current_metadata = []
        
        # Process kinematic values from frame texture movement
        speed = self._calculate_optical_flow_speed(frame)
        
        curr_time = time.time()
        dt = curr_time - self.prev_time
        if dt > 0.01:
            distance_meters = (speed / 3.6) * dt
            mock_lat = self.prev_lat + (distance_meters / 111139.0)
            mock_lon = self.prev_lon
            self.prev_lat, self.prev_lon, self.prev_time = mock_lat, mock_lon, curr_time
        else:
            mock_lat, mock_lon = self.prev_lat, self.prev_lon

        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        address = self._get_offline_address(mock_lat, mock_lon)

        # Draw Collision Mitigation Intrusion Zones
        intrusion_zone = [int(self.frame_width * 0.25), int(self.frame_height * 0.55), int(self.frame_width * 0.75), self.frame_height]
        overlay = frame.copy()
        cv2.rectangle(overlay, (intrusion_zone[0], intrusion_zone[1]), (intrusion_zone[2], intrusion_zone[3]), (255, 0, 0), -1)
        cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

        results = self.model(frame, verbose=False)[0]
        min_conf = self.config.get("min_confidence", 0.45)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls_id in self.CLASS_MAPPINGS and conf > min_conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                obj_name = self.CLASS_MAPPINGS[cls_id]
                
                pixel_height = max(1, y2 - y1)
                real_height = self.REAL_HEIGHTS.get(obj_name, 1.5)
                estimated_distance = (real_height * self.FOCAL_LENGTH_PIXELS) / pixel_height

                is_intruding = (x2 > intrusion_zone[0] and x1 < intrusion_zone[2] and y2 > intrusion_zone[1])
                
                self.current_metadata.append({
                    "category_id": cls_id, "name": obj_name, "confidence": conf,
                    "bbox": [x1, y1, x2, y2], "distance_meters": round(estimated_distance, 2),
                    "intrusion_alert": is_intruding, "gps": {"lat": mock_lat, "lon": mock_lon},
                    "speed_kmh": speed, "direction": "Northbound (N)", "address": address, "timestamp": timestamp_str
                })

                box_color = (0, 0, 255) if is_intruding else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                ui_label = f"{obj_name} | {estimated_distance:.1f}m"
                cv2.putText(frame, ui_label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv2.LINE_AA)

        # Draw HUD Dashboard Telemetry Ribbon
        hud_bg = frame.copy()
        cv2.rectangle(hud_bg, (0, 0), (self.frame_width, 42), (0, 0, 0), -1)
        cv2.addWeighted(hud_bg, 0.65, frame, 0.35, 0, frame)
        
        telemetry_txt = f"{timestamp_str} | {speed} km/h (Optical Flow) | Northbound (N)"
        cv2.putText(frame, telemetry_txt, (12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        geo_txt = f"LOC: {address} | LAT: {mock_lat:.5f} | LON: {mock_lon:.5f}"
        cv2.putText(frame, geo_txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        self.latest_annotated_frame = self._apply_hud(frame)
        self.pre_buffer.append(self.latest_annotated_frame.copy())

        if self.writer is not None:
            self.writer.write(self.latest_annotated_frame)

        now = time.monotonic()
        if (now - self._last_geocode_lookup) >= self._geocode_interval_sec:
            self.current_address = self._get_offline_address(mock_lat, mock_lon)
            self._last_geocode_lookup = now

        return True

    def get_ai_metadata(self) -> list:
        """Concrete implementation required by BaseCamera interface."""
        return self.current_metadata

    def save_incident_snapshot(self, output_path: str, hazard_info: dict = None) -> None:
        if self.latest_annotated_frame is not None:
            cv2.imwrite(output_path, self.latest_annotated_frame)

    def start_recording(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            float(self.target_fps),
            (int(self.frame_width), int(self.frame_height)),
        )

    def stop_recording(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def get_latest_frame(self):
        return self.latest_annotated_frame

    def write_pre_buffer_to_incident(self, incident_clip_path: str) -> None:
        os.makedirs(os.path.dirname(incident_clip_path), exist_ok=True)
        self.start_recording(incident_clip_path)
        for frame in self.pre_buffer:
            self.writer.write(frame)

    def close(self) -> None:
        """Release all camera/codec resources safely."""
        try:
            self.stop_recording()
        except Exception:
            pass

        if getattr(self, "cap", None) is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass