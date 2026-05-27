# Transfer Script

This folder contains `transfer_to_ground.ps1`, a PowerShell script that pulls a ground run bundle from the Jetson to your Windows machine.

## What It Copies

The script collects these remote paths:

- `ML-classifications/camera-pipeline/images`
- `voiceNotes/data`
- `led-display/sage_data`
- `/data/sage`
- `log.txt`

It saves everything into a timestamped folder under `Downloads\sage_ground_data` by default.

## Requirements

- Windows PowerShell
- OpenSSH client installed on Windows
- SSH access to the Jetson

## How To Use

1. Open PowerShell in this folder or copy the script to your working directory.
2. Run the script with your Jetson username and host, or let it prompt you for them.
3. The script will test SSH connectivity unless you pass `-SkipConnectionTest`.

Example:

```powershell
.\transfer_to_ground.ps1 -JetsonUser jetson -JetsonHost 192.168.0.22
```

## Commands To Paste

Paste your setup commands below:

```powershell
scp jetson@192.168.0.22:/home/jetson/CapstoneGit/capstone/scripts/transfer_to_ground.ps1 .
powershell -ExecutionPolicy Bypass -File .\transfer_to_ground.ps1
```

## Notes

- Use `-DestRoot` if you want to save the output somewhere other than `Downloads\sage_ground_data`.
- Use `-Port` if SSH is not running on the default port `22`.
- Use `-SkipConnectionTest` if you already know the connection works and want to skip the initial SSH check.