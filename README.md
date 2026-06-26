# AI Dashcam

Local-first AI dashcam for macOS and Raspberry Pi-style camera workflows.  
It supports video playback, live capture, object detection, telemetry overlays, and incident snapshot/clip logging.

## Features

- Real-time video processing with OpenCV
- YOLO-based object detection
- Optical-flow speed estimation
- GPS/address metadata simulation
- Incident snapshots and buffered clip recording
- Frame pacing with configurable FPS
- Local HTTP media portal for recorded incidents

## Project Structure

- `src/main.py` — application entry point and orchestration
- `src/camera/mac_camera.py` — macOS camera / video source pipeline
- `src/camera/pi_camera.py` — Pi camera / generic camera pipeline
- `src/camera/base_camera.py` — shared camera interface
- `src/processing/analytics.py` — hazard and threat analysis
- `src/storage/circular_buffer.py` — storage retention management
- `docs/` — architecture, developer, hardware, and software requirements

## Requirements

- Python 3.14
- macOS or Linux
- OpenCV
- NumPy
- Ultralytics
- reverse-geocoder

## Install

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

## Configuration

The main runtime configuration is in `src/main.py`.

Key settings:

- `storage.clip_dir` — directory for clips and incident assets
- `storage.max_storage_mb` — storage retention limit
- `storage.clip_duration_seconds` — recording segment length
- `analytics.min_confidence` — detection threshold
- `analytics.video_source` — video file path or camera source
- `network.bind_address` — local media server bind address
- `network.port` — local media server port
- `fps` — structural frame rate for playback and recording

## FPS and Playback Control

The application uses a structural FPS rate to prevent fast playback.

- `camera.fps` is used as the source FPS
- recording uses the same FPS value in `cv2.VideoWriter(...)`
- the main loop uses `time.monotonic()` and `time.sleep()` to maintain real-time pacing

If video appears too fast:
- lower `fps` in config
- ensure the source video FPS is detected correctly
- verify the main loop is sleeping for `1 / fps` seconds per frame

## Incident Logging

When a threat is detected, the system creates a dedicated incident folder:

```text
incident_YYYYMMDD_HHMMSS_mmm
```

Inside each incident folder:

- `snapshot_<timestamp>.jpg`
- `clip_<timestamp>.avi`

The timestamp format is:

```text
YYYYMMDD_HHMMSS_mmm
```

This keeps snapshot and clip files synchronized.

## Documentation

- [Architecture Overview](./docs/architecture.md)
- [Developer Guide](./docs/developer-guide.md)
- [Hardware Requirements](./docs/hardware-requirements.md)
- [Software Requirements](./docs/software-requirements.md)

## Notes

- The app is designed to prefer local assets first.
- If the video source cannot be opened, it falls back to live webcam input.
- Press `q` in the video window to stop the camera loop.