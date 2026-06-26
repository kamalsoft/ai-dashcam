# AI-Dashcam

## Execute
python3 -m venv path/to/venv
source path/to/venv/bin/activate
pip3 install -r requirements-dev.txt



## Create the Hotspot via Command Line
sudo nmcli device wifi hotspot ifname wlan0 ssid "Dashcam_Pro_AP" password "NanovianADAS2026"

2. Lock Hotspot to Auto-Start on System Boot
By default, NetworkManager will drop the hotspot if it sees a known home Wi-Fi network. For a dedicated dashcam environment, we want the system to force-broadcast its own access point as soon as the ignition turns on.

Run this command to find the unique connection profile metadata:
```bash
nmcli connection show
```

Look for the line containing Dashcam_Pro_AP and copy its UUID or connection name. Then execute these property adjustments:

Bash
# Force the connection profile to spin up automatically on every boot cycle
```bash
sudo nmcli connection modify "Hotspot" connection.autoconnect yes
sudo nmcli connection modify "Hotspot" connection.autoconnect-priority 100
```
Once configured, your dashcam hardware will host its own internal server at the gateway IP address: 192.168.4.1.