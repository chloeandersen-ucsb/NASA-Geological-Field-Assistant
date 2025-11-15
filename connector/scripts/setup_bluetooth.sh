#!/usr/bin/env bash
set -euo pipefail

echo "[setup] Installing Bluetooth stack & helpers"
sudo apt-get update
sudo apt-get install -y bluetooth bluez bluez-tools python3-venv python3-pip linux-firmware rfkill

echo "[setup] Ensure bluetoothd runs with --experimental (GATT over D-Bus)"
if ! grep -q -- "--experimental" /lib/systemd/system/bluetooth.service; then
  sudo sed -i 's|^ExecStart=.*|ExecStart=/usr/lib/bluetooth/bluetoothd --experimental|' /lib/systemd/system/bluetooth.service || \
  sudo sed -i 's|^ExecStart=.*|ExecStart=/usr/sbin/bluetoothd --experimental|' /lib/systemd/system/bluetooth.service || true
  sudo systemctl daemon-reload
fi

echo "[setup] Restart bluetoothd"
sudo systemctl restart bluetooth
sudo rfkill unblock bluetooth || true
sleep 1

echo "[setup] Adapter info:"
echo -e 'power on\nshow\nquit' | sudo bluetoothctl || true

cat << 'EONOTE'

USB-BT500 (Realtek RTL8761B) notes:
- If 'bluetoothctl show' doesn't list hci0, re-plug the dongle and check:
    dmesg | egrep -i 'rtl|btusb|blue|firmware'
- Make sure 'linux-firmware' is installed and up to date.
- Reboot if firmware just got installed.
EONOTE
