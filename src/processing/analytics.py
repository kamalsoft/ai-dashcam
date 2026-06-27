# src/processing/analytics.py
 import logging
import time

logger = logging.getLogger("DashcamCore")

class ThreatAnalytics:
    def __init__(self, config: dict):
        """
        Initializes the intelligence analytics post-processor.
        
        Expects a config block looking for 'min_confidence' and analytical limits.
        """
        # Read from config blocks, fallback gracefully to mathematical defaults
        self.min_alert_confidence = float(config.get("min_confidence", 0.45))
        self.rapid_expansion_threshold = float(config.get("rapid_expansion_threshold", 1.25))
        
        # In-memory tracking cache tracking moving objects frame-over-frame
        # Format: { track_id: { "last_area": float, "last_seen": float } }
        self.tracking_history = {}
        
        # Track targeted threat classes from standard YOLO metadata profiles (e.g., car, truck, bus, person)
        self.target_classes = {0, 2, 3, 5, 7} 

    def calculate_box_area(self, bbox: list) -> float:
        """Calculates exact area of a bounding box format vector: [x1, y1, x2, y2]"""
        x1, y1, x2, y2 = bbox
        width = max(0.0, float(x2 - x1))
        height = max(0.0, float(y2 - y1))
        return width * height

    def process_inference_metadata(self, frame_metadata: list) -> bool:
        """
        Evaluates active tracking matrices frame-over-frame.
        Returns True immediately if an imminent threat/rapid proximity expansion is confirmed.
        """
        if not frame_metadata:
            return False

        current_time = time.monotonic()
        threat_detected = False

        # Prune old stale tracking keys out of RAM if they haven't been seen in over 2 seconds
        stale_ids = [tid for tid, data in self.tracking_history.items() if current_time - data["last_seen"] > 2.0]
        for tid in stale_ids:
            del self.tracking_history[tid]

        for detection in frame_metadata:
            if not isinstance(detection, dict):
                continue
                
            # Direct shortcut bypass if another module upstream already flag a critical boundary violation
            if detection.get("intrusion_alert", False):
                return True

            bbox = detection.get("box")        # Format expected: [x1, y1, x2, y2]
            conf = detection.get("conf", 0.0)
            cls_id = detection.get("cls", -1)
            track_id = detection.get("track_id") # Requires YOLO tracking enabled (.track() instead of .predict())

            # Filter operations: Is this a high-confidence target vehicle class?
            if conf < self.min_alert_confidence or cls_id not in self.target_classes:
                continue

            if bbox and track_id is not None:
                current_area = self.calculate_box_area(bbox)
                
                if track_id in self.tracking_history:
                    historical_data = self.tracking_history[track_id]
                    previous_area = historical_data["last_area"]
                    
                    if previous_area > 0:
                        # Compute relative area scaling delta ratio
                        expansion_ratio = current_area / previous_area
                        
                        # If a vehicle's frame presence swells beyond our safety limit, trigger containment
                        if expansion_ratio >= self.rapid_expansion_threshold:
                            logger.warning(
                                f"‼️ COLLISION THREAT: Track ID {track_id} expanded rapidly by "
                                f"{((expansion_ratio - 1) * 100):.1f}% over frame updates!"
                            )
                            threat_detected = True
                    
                    # Update cache entry state
                    self.tracking_history[track_id] = {
                        "last_area": current_area,
                        "last_seen": current_time
                    }
                else:
                    # Seed entry into persistent RAM tracking cache
                    self.tracking_history[track_id] = {
                        "last_area": current_area,
                        "last_seen": current_time
                    }

        return threat_detected