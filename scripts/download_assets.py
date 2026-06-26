# scripts/download_assets.py
import os
from ultralytics import YOLO

def cache_assets_locally():
    target_dir = "./models"
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "yolov8n.pt")
    
    if not os.path.exists(target_path):
        print("[Build Phase] Downloading local model weights to cache repository...")
        # Pull down the base weights cleanly over standard SSL streams
        model = YOLO("yolov8n.pt")
        # Move the asset to our persistent local directory
        os.rename("yolov8n.pt", target_path)
        print("[Build Phase] Native weight files safely cached to disk.")
    else:
        print("[Build Phase] Existing local weights found inside target directory.")
        
    print("[Build Phase] All assets locally locked down and ready for vehicle deployment.")

if __name__ == "__main__":
    cache_assets_locally()