# Architecture Overview

## System Goal
AI Dashcam is a local-first dashcam application for macOS and Raspberry Pi-style camera workflows. It supports live video ingestion, object detection, telemetry overlay, incident logging, and local clip access.

## High-Level Flow
1. Camera source is initialized.
2. Frames are captured at a structural FPS rate.
3. AI analytics run on each frame.
4. Threats trigger incident logging.
5. Snapshots and clips are stored locally.
6. A local HTTP portal exposes recorded assets.

## Core Modules

### `src/main.py`
- Application entry point
- Runtime orchestration
- Frame pacing control
- Incident lifecycle management
- Local media portal startup

### `src/camera/base_camera.py`
- Shared camera interface
- Common recording and snapshot contract

### `src/camera/mac_camera.py`
- macOS video source pipeline
- Webcam fallback
- YOLO detection
- Optical flow speed estimation
- Metadata generation

### `src/camera/pi_camera.py`
- Generic or Raspberry Pi camera pipeline
- Shared camera behavior
- Recording support

### `src/processing/analytics.py`
- Threat inference and hazard evaluation
- Metadata interpretation

### `src/storage/circular_buffer.py`
- Storage retention enforcement
- Clip cleanup based on capacity rules

## Incident Logging Design
When a threat is detected:
- A dedicated incident directory is created
- Snapshot and clip share the same timestamp
- Assets are stored together for traceability

Directory format:
```text
incident_YYYYMMDD_HHMMSS_mmm
```

Files:
- `snapshot_<timestamp>.jpg`
- `clip_<timestamp>.avi`

## FPS Strategy
The application uses a structural FPS value to avoid fast playback:
- Source FPS is detected from the camera or video file
- `cv2.VideoWriter(...)` uses the same FPS
- Main loop pacing uses `time.monotonic()` and `time.sleep()`

## Local-First Principles
- Prefer local video assets
- Fall back to live camera input
- Avoid cloud dependencies for core runtime
- Keep storage and incident artifacts local