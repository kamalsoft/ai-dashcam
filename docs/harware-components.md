# Raspberry Pi Hardware Components (AI Dashcam)

## Required Core Components

- **Raspberry Pi 5 (recommended)** or **Raspberry Pi 4 Model B (minimum 4GB RAM)**
- **microSD card** (minimum 32GB, recommended 64GB+ UHS-I A2)
- **Raspberry Pi Camera Module** (v2/v3) or **UVC USB webcam**
- **Stable power supply**
  - Pi 5: official 27W USB-C PSU recommended
  - Pi 4: official 5V/3A PSU recommended
- **Cooling solution**
  - Passive heatsink minimum
  - Active fan recommended for continuous YOLO inference

## Storage & Recording

- **Local storage for clips/snapshots**
  - Minimum: 5GB free
  - Recommended: external USB 3.0 SSD for long-duration recording
- **Sustained write speed** is important for AVI clip generation and incident logging

## Optional but Recommended

- **GPS module** (if replacing simulated location)
- **IMU sensor** (for improved motion/heading estimation)
- **LTE/Wi-Fi hotspot dongle** (remote access scenarios)
- **12V to USB-C car power adapter** (in-vehicle deployment)

## Camera Interface Notes

- **CSI camera**: lower CPU overhead and better integration
- **USB webcam**: easier setup, broader compatibility
- Ensure fixed mounting and vibration control for better optical-flow stability

## Performance Targets (Pi Deployment)

- **Resolution:** 640x480 or 1280x720 for stable real-time inference
- **FPS target:** 15–20 FPS on Pi 4/5 depending on model load
- Keep `target_fps` aligned across capture loop and `VideoWriter` to avoid fast playback artifacts

## Minimum Deployment Checklist

- [ ] Pi boots reliably with Python 3.14 environment
- [ ] Camera accessible via OpenCV (`cv2.VideoCapture`)
- [ ] Storage path writable (`mock_dashcam_clips/`)
- [ ] Incident folder creation working (`incident_YYYYMMDD_HHMMSS_mmm`)
- [ ] Thermal throttling avoided under continuous runtime