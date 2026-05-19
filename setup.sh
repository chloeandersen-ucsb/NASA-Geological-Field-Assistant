#!/bin/bash
# Run this once on the Jetson. It installs itself to run automatically on every
# power-on and wake from sleep, then launches the app immediately.

# to install on jetson:
# cd /path/to/capstone
# sudo ./setup.sh --install

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME=$(eval echo "~$REAL_USER")
AUTOSTART_FILE="$REAL_HOME/.config/autostart/sage.desktop"
SLEEP_HOOK="/lib/systemd/system-sleep/sage-wakeup"

is_jetson() {
    [[ -f /etc/nv_tegra_release ]]
}

rotate_screen() {
    local attempts=0
    until xrandr &>/dev/null || (( attempts >= 15 )); do
        sleep 1
        (( attempts++ )) || true
    done

    local output
    output=$(xrandr 2>/dev/null | awk '/ connected/{print $1; exit}')

    if [[ -z "$output" ]]; then
        echo "[sage] WARNING: no connected display found, skipping rotation" >&2
        return
    fi

    local min_dim_mm
    min_dim_mm=$(xrandr 2>/dev/null | grep ' connected' | grep -o '[0-9]*mm x [0-9]*mm' | \
        awk -F'[^0-9]+' '{w=$1+0; h=$2+0; print (w<h?w:h)}')

    if [[ "$output" == DSI* ]] || { [[ -n "$min_dim_mm" ]] && (( min_dim_mm < 120 )); }; then
        echo "[sage] Small display detected ($output, min dim: ${min_dim_mm:-unknown}mm) — rotating inverted"
        xrandr --output "$output" --rotate inverted
    else
        echo "[sage] Large display detected ($output, min dim: ${min_dim_mm:-unknown}mm) — skipping rotation"
    fi
}

install_autostart() {
    mkdir -p "$HOME/.config/autostart"
    cat > "$AUTOSTART_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=SAGE
Comment=SAGE startup — screen rotation and application launch
Exec=$SCRIPT_DIR/setup.sh
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
EOF
    echo "[sage] Installed autostart entry: $AUTOSTART_FILE"
}

install_sleep_hook() {
    sudo tee "$SLEEP_HOOK" > /dev/null <<EOF
#!/bin/bash
# Invoked by systemd-sleep on suspend/resume events.
[[ "\$1" == "post" ]] || exit 0

ACTIVE_USER=\$(loginctl list-sessions --no-legend 2>/dev/null | awk '{print \$3}' | head -1)
[[ -z "\$ACTIVE_USER" ]] && ACTIVE_USER="$USER"
USER_HOME=\$(eval echo "~\$ACTIVE_USER")

sudo -u "\$ACTIVE_USER" \
    DISPLAY=":0" \
    XAUTHORITY="\$USER_HOME/.Xauthority" \
    bash "$SCRIPT_DIR/setup.sh" &
EOF
    sudo chmod +x "$SLEEP_HOOK"
    echo "[sage] Installed wake hook: $SLEEP_HOOK"
}

install_if_needed() {
    local installed=true

    if [[ ! -f "$AUTOSTART_FILE" ]]; then
        install_autostart
        installed=false
    fi

    if [[ ! -f "$SLEEP_HOOK" ]]; then
        install_sleep_hook
        installed=false
    fi

    if [[ "$installed" == false ]]; then
        echo "[sage] First-time setup complete. Will auto-run on every power-on and wake."
    fi
}

main() {
    if ! is_jetson; then
        echo "[sage] Not a Jetson device — nothing to do."
        exit 0
    fi

    install_if_needed
    rotate_screen
    cd "$SCRIPT_DIR"
    exec sudo -u "${SUDO_USER:-$USER}" make run
}

main "$@"
