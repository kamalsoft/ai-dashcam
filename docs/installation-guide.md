# Installation Guide (Raspberry Pi Priority)

This guide is for **Raspberry Pi production deployment**.  
macOS is only for development/testing.

## 1) Hardware + OS Prerequisites

- Raspberry Pi 5 (recommended) or Pi 4 (4GB+ RAM)
- Raspberry Pi Camera Module (CSI) or USB UVC webcam
- 32GB+ microSD (64GB+ recommended) or USB SSD
- Raspberry Pi OS 64-bit (Bookworm recommended)
- Stable power supply + cooling

---

## 2) System Preparation (Raspberry Pi)

Update OS and install required system packages:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3 python3-venv python3-pip libatlas-base-dev libjpeg-dev libopenjp2-7 ffmpeg
```

If using a CSI camera, enable camera support and reboot:

```bash
sudo raspi-config
# Interface Options -> Camera -> Enable
sudo reboot
```

Quick camera sanity check (after reboot):

```bash
libcamera-hello
```

---

## 3) Clone Project

```bash
cd ~
git clone https://github.com/kamalsoft/ai-dashcam.git
cd ai-dashcam
```

---

## 4) Create Python Environment

Use your project’s required Python version.  
If Python 3.14 is unavailable on Pi OS, use the highest supported version and pin dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 5) Download YOLO Model (Required)

Create model directory and download `yolov8n.pt`:

```bash
mkdir -p models
wget -O models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
```

Alternative (Ultralytics auto-download):

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt models/yolov8n.pt 2>/dev/null || true
```

Expected final path:

```text
./models/yolov8n.pt
```

---

## 6) Configure for Pi Runtime

In `src/main.py` config, ensure:

- `platform: "pi"`
- valid source in `analytics.video_source` (or camera index/device)
- writable output path (`mock_dashcam_clips/`)

Suggested values:

- `storage.clip_dir = "./mock_dashcam_clips"`
- `analytics.fps = 15.0` to `20.0` for Pi stability

---

## 7) Run Application

```bash
source .venv/bin/activate
python -u src/main.py
```

---

## 8) Verify Outputs

- Normal clips: `mock_dashcam_clips/clip_*.avi`
- Incident folders:
  - `mock_dashcam_clips/incident_YYYYMMDD_HHMMSS_mmm/`
  - `snapshot_<timestamp>.jpg`
  - `clip_<timestamp>.avi`

---

## 9) Optional: Run on Boot (systemd)

Preferred (uses bundled installer script):

```bash
cd ~/ai-dashcam
sudo bash scripts/setup_systemd.sh
```

This creates and enables `ai-dashcam.service` so the app starts automatically on every Pi boot.

Manual option:

Create service file:

```bash
sudo nano /etc/systemd/system/ai-dashcam.service
```

Use:

```ini
[Unit]
Description=AI Dashcam Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/ai-dashcam
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/pi/ai-dashcam/.venv/bin/python -u /home/pi/ai-dashcam/src/main.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-dashcam
sudo systemctl start ai-dashcam
sudo systemctl status ai-dashcam
```

View logs:

```bash
journalctl -u ai-dashcam -f
```

---

## 10) Pi Troubleshooting

### Camera not opening

- Run `libcamera-hello`
- Verify camera interface is enabled in `raspi-config`
- Confirm device visibility:

  ```bash
  ls /dev/video*
  ```

### Slow inference

- Lower resolution
- Reduce FPS to 15
- Ensure active cooling

### Fast/slow playback

- Keep `target_fps`, frame pacing, and `cv2.VideoWriter` FPS aligned

### Disk fills quickly

- Reduce clip duration/FPS
- Verify retention policy is active
