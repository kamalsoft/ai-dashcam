#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="ai-dashcam.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

if [[ "${EUID}" -ne 0 ]]; then
	echo "Please run as root: sudo bash scripts/setup_systemd.sh"
	exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_USER="${SUDO_USER:-$(logname 2>/dev/null || true)}"
if [[ -z "${APP_USER}" || "${APP_USER}" == "root" ]]; then
	APP_USER="pi"
fi

APP_GROUP="$(id -gn "${APP_USER}" 2>/dev/null || echo "${APP_USER}")"
PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
	PYTHON_BIN="/usr/bin/python3"
fi

echo "Installing ${SERVICE_NAME}"
echo "  Repo:   ${REPO_DIR}"
echo "  User:   ${APP_USER}:${APP_GROUP}"
echo "  Python: ${PYTHON_BIN}"

cat > "${SERVICE_PATH}" <<EOF
[Unit]
Description=AI Dashcam Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${APP_USER}
Group=${APP_GROUP}
WorkingDirectory=${REPO_DIR}
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=${REPO_DIR}
ExecStart=${PYTHON_BIN} -m src.main
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"

echo "${SERVICE_NAME} installed and started."
echo "Check status: sudo systemctl status ${SERVICE_NAME}"
echo "Follow logs:  sudo journalctl -u ${SERVICE_NAME} -f"
