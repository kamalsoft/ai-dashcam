import logging
import time
import math

logger = logging.getLogger("DashcamCore")

class ThreatAnalytics:
    def __init__(self, config: dict):
        self.min_alert_confidence = float(config.get("min_confidence", 0.40))
        self.expansion_velocity_threshold = 0.85
        self.tracking_history = {}
        self.target_classes = {2, 3, 5, 7}  # Car, motorcycle, bus, truck
        self.threat_streak = {}
        
        # Real-world spatial calibrations (assuming camera is mounted at ~4.5 feet high)
        self.KNOWN_CAM_WIDTH_PX = 640
        self.KNOWN_CAM_HEIGHT_PX = 480
        
        # In-memory speed cache { track_id: [list_of_recent_speeds] } for rolling average smoothing
        self.speed_history = {}

    def calculate_box_area(self, bbox: list) -> float:
        x1, y1, x2, y2 = bbox
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    def estimate_speed_mph(self, bbox: list, dt: float, track_id: int) -> float:
        """
        Estimates relative speed by tracking the camera distance change over time.
        Uses the inverse relationship of bounding box height to distance.
        """
        x1, y1, x2, y2 = bbox
        box_height = max(1.0, float(y2 - y1))
        
        # Empirical scaling factor converting pixel dimensions to approximate distance
        # Focal length approximation factor for standard widescreen dashcam fields of view
        focal_distance_constant = 1100.0 
        current_distance_feet = focal_distance_constant / box_height
        
        historical_data = self.tracking_history.get(track_id)
        if not historical_data or "last_bbox" not in historical_data:
            return 0.0
            
        _, _, _, old_y2 = historical_data["last_bbox"]
        old_height = max(1.0, float(old_y2 - historical_data["last_bbox"][1]))
        prior_distance_feet = focal_distance_constant / old_height
        
        # Calculate velocity component
        distance_delta_feet = prior_distance_feet - current_distance_feet
        speed_fps = distance_delta_feet / dt
        speed_mph = speed_fps * 0.681818 # Convert feet per second to MPH
        
        # Filter out extreme values caused by single-frame tracking glitches
        if abs(speed_mph) > 120.0:
            return 0.0 if not self.speed_history.get(track_id) else self.speed_history[track_id][-1]
            
        # Smooth with a rolling average filter over the last 5 tracking loops
        if track_id not in self.speed_history:
            self.speed_history[track_id] = []
        self.speed_history[track_id].append(speed_mph)
        if len(self.speed_history[track_id]) > 5:
            self.speed_history[track_id].pop(0)
            
        return sum(self.speed_history[track_id]) / len(self.speed_history[track_id])

    def process_inference_metadata(self, frame_metadata: list) -> bool:
        if not frame_metadata:
            return False

        current_time = time.monotonic()
        system_lockout_triggered = False
        active_track_ids = set()

        for detection in frame_metadata:
            if not isinstance(detection, dict):
                continue
                
            if detection.get("intrusion_alert", False):
                return True

            bbox = detection.get("box")        
            conf = detection.get("conf", 0.0)
            cls_id = detection.get("cls", -1)
            track_id = detection.get("track_id") 

            if conf < self.min_alert_confidence or cls_id not in self.target_classes:
                continue

            if bbox and track_id is not None:
                active_track_ids.add(track_id)
                x1, y1, x2, y2 = bbox
                box_center_x = (x1 + x2) / 2.0
                
                # Keep wide field of view window checking active
                if box_center_x < 60 or box_center_x > 580:
                    continue

                current_area = self.calculate_box_area(bbox)
                calculated_speed = 0.0
                
                if track_id in self.tracking_history:
                    historical_data = self.tracking_history[track_id]
                    previous_area = historical_data["last_area"]
                    dt = current_time - historical_data["last_seen"]
                    
                    if previous_area > 0 and dt > 0.001:
                        # Append calculated speed to dictionary object for rendering engine context access
                        calculated_speed = self.estimate_speed_mph(bbox, dt, track_id)
                        detection["speed_mph"] = calculated_speed
                        
                        area_delta_ratio = (current_area - previous_area) / previous_area
                        expansion_velocity = area_delta_ratio / dt
                        
                        if expansion_velocity >= self.expansion_velocity_threshold:
                            self.threat_streak[track_id] = self.threat_streak.get(track_id, 0) + 1
                            if self.threat_streak[track_id] >= 3:
                                system_lockout_triggered = True
                        else:
                            self.threat_streak[track_id] = max(0, self.threat_streak.get(track_id, 0) - 1)
                    
                    self.tracking_history[track_id] = {
                        "last_area": current_area, 
                        "last_seen": current_time,
                        "last_bbox": bbox
                    }
                else:
                    self.tracking_history[track_id] = {
                        "last_area": current_area, 
                        "last_seen": current_time,
                        "last_bbox": bbox
                    }
                    self.threat_streak[track_id] = 0
                    detection["speed_mph"] = 0.0

        # Prune memory caches
        stale_streaks = [tid for tid in list(self.threat_streak.keys()) if tid not in active_track_ids]
        for tid in stale_streaks:
            del self.threat_streak[tid]
            if tid in self.speed_history:
                del self.speed_history[tid]

        return system_lockout_triggered