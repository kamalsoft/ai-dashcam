# src/processing/analytics.py
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class ThreatAnalytics:
    def __init__(self, rapid_expansion_threshold: float = 1.25, min_alert_confidence: float = 0.60):
        """
        Initializes the intelligence analytics post-processor.
        
        :param rapid_expansion_threshold: Multiplier indicating dangerous bounding box growth rate.
        :param min_alert_confidence: Minimum AI model accuracy score to consider for threats.
        """
        self.rapid_expansion_threshold = rapid_expansion_threshold
        self.min_alert_confidence = min_alert_confidence
        
        # In-memory tracking cache. Maps tracking IDs or position hashes to past box areas.
        # Format: { category_id: { "last_area": float, "last_seen": float } }
        self.tracking_history = {}

    def calculate_box_area(self, bbox: list) -> float:
        """Calculates area of bounding box: [x1, y1, x2, y2]"""
        x1, y1, x2, y2 = bbox
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        return float(width * height)

    def process_inference_metadata(self, frame_metadata):
        if not frame_metadata:
            return False

        for item in frame_metadata:
            if isinstance(item, dict) and item.get("intrusion_alert", False) is True:
                return True

            boxes = item.get("boxes") if isinstance(item, dict) else None
            if isinstance(boxes, list):
                for b in boxes:
                    if isinstance(b, dict) and b.get("intrusion_alert", False) is True:
                        return True

        return False