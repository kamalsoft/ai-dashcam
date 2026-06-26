# Software Requirements

## Runtime
- Python 3.14

## Operating Systems
- macOS
- Linux
- Raspberry Pi OS compatible environments

## Python Packages
- OpenCV
- NumPy
- Ultralytics
- reverse-geocoder

## Suggested Environment Setup
```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Application Dependencies
- Video capture support via OpenCV
- YOLO model weights available locally
- Local filesystem access for clips and snapshots
- Optional local HTTP server support

## Model Asset
The application expects a local YOLO model file:
- `./models/yolov8n.pt`

## File System Requirements
The app must be able to:
- create directories
- write JPG snapshots
- write AVI clips
- delete old media during retention cleanup

## Configuration Requirements
Runtime config should define:
- `storage.clip_dir`
- `storage.max_storage_mb`
- `storage.clip_duration_seconds`
- `analytics.min_confidence`
- `analytics.video_source`
- `network.bind_address`
- `network.port`
- `fps` where applicable

## Compatibility Notes
- OpenCV must support the selected camera or file backend
- `cv2.VideoWriter` must be available for clip creation
- FFmpeg support may be required depending on the OpenCV build

## Developer Tools
Recommended:
- Visual Studio Code
- Python extension
- Integrated terminal
- Git