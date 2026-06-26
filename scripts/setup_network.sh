#!/bin/bash
# scripts/setup_network.sh

SSID="Dashcam_Pro_AP"
PASSWORD="NanovianADAS2026"
INTERFACE="wlan0"

echo "[Network Setup] Checking operating system environment..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "──────────────────────────────────────────────────────────────"
    echo "[!] macOS Detected (MacBook Testbed)"
    echo "──────────────────────────────────────────────────────────────"
    echo "Linux 'nmcli' tooling is not supported on native macOS kernels."
    echo "To simulate a local access point on your development box:"
    echo "  1. Go to System Settings -> Sharing -> Internet Sharing."
    echo "  2. Turn it on and configure your Wi-Fi hotspot options."
    echo "──────────────────────────────────────────────────────────────"
    exit 0
fi

# Process provisioning if running on real Linux hardware
if command -v nmcli &> /dev/null; then
    echo "[+] NetworkManager found. Provisioning hardware Access Point..."
    
    # Bring down any conflicting active connections on wlan0
    sudo nmcli device disconnect $INTERFACE 2>/dev/null
    
    # Build out the native wireless hotspot instance
    sudo nmcli device wifi hotspot ifname $INTERFACE ssid "$SSID" password "$PASSWORD"
    
    # Configure profile to auto-initialize on bootup
    sudo nmcli connection modify "Hotspot" connection.autoconnect yes
    sudo nmcli connection modify "Hotspot" connection.autoconnect-priority 100
    
    echo "[+] Wireless Host Hotspot successfully established on $INTERFACE."
    echo "    SSID: $SSID"
    echo "    Portal Gateway Endpoint: http://192.168.4.1:8080/"
else
    echo "[-] Error: 'nmcli' not found. Ensure NetworkManager is installed on your Linux device."
    exit 1
fi