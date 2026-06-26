# Developer Guide

## Purpose
This project provides a local-first AI dashcam pipeline with incident capture, playback pacing, and media export.

## Repository Layout
- `src/main.py` — app orchestration
- `src/camera/` — camera implementations
- `src/processing/` — threat analytics
- `src/storage/` — retention utilities
- `docs/` — architecture and requirements documents

## Running the App
```bash
python src/main.py
```

## Common Development Tasks

### Adjust FPS
Update:
- `camera.fps`
- video writer FPS
- frame pacing logic in `src/main.py`

### Update Incident Behavior
Edit:
- incident directory naming
- snapshot filename format
- clip buffering duration
- post-threat cooldown window

### Add a Camera Backend
Implement `BaseCamera` in a new file under `src/camera/`.

Required methods:
- `initialize(config: dict)`
- `update_frame() -> bool`
- `get_ai_metadata() -> list`
- `start_recording(output_path: str, fps: float | None = None)`
- `stop_recording()`
- `save_incident_snapshot(...)`
- `close()`

## Debug Tips
- Verify the input source path exists
- Confirm `camera.fps` matches the intended playback rate
- Check that `VideoWriter` is released after each incident
- Ensure output directories exist before writing files
- Use the local HTTP portal to inspect generated assets

## Incident Workflow
1. Threat is detected
2. Timestamp is frozen immediately
3. Snapshot is written
4. Clip writer is started or reused
5. Buffer frames are flushed into the clip
6. Writer is released on incident completion

## Coding Notes
- Prefer `time.monotonic()` for loop timing
- Avoid hardcoded frame sleeps
- Release OpenCV resources in `close()`
- Keep file paths configurable through `APP_CONFIG`