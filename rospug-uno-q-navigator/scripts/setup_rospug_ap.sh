#!/usr/bin/env bash
set -euo pipefail

SSID="${1:-ROSPug-Pro}"
PASSWORD="${2:-ROSPugDemo2026}"
BAND_REQUEST="${3:-2.4}"
CONNECTION_NAME="ROSPug-AP"
AP_ADDRESS="10.42.0.1/24"

if (( ${#PASSWORD} < 8 || ${#PASSWORD} > 63 )); then
  echo "The WPA password must contain 8 to 63 characters." >&2
  exit 2
fi

case "${BAND_REQUEST,,}" in
  2.4|2.4ghz|2g|bg)
    NM_BAND="bg"
    BAND_LABEL="2.4 GHz"
    ;;
  5|5ghz|5g|a)
    NM_BAND="a"
    BAND_LABEL="5 GHz"
    ;;
  *)
    echo "Unsupported band '$BAND_REQUEST'. Use '2.4' or '5'." >&2
    exit 2
    ;;
esac

WIFI_INTERFACE="$(nmcli -t -f DEVICE,TYPE device status | awk -F: '$2 == "wifi" {print $1; exit}')"
if [[ -z "$WIFI_INTERFACE" ]]; then
  echo "No NetworkManager Wi-Fi interface was found." >&2
  exit 3
fi

CURRENT_CONNECTION="$(nmcli -g GENERAL.CONNECTION device show "$WIFI_INTERFACE" 2>/dev/null || true)"

cat <<EOF
ROSPug access-point setup
  Interface:          $WIFI_INTERFACE
  Current connection: ${CURRENT_CONNECTION:---}
  New Wi-Fi network:  $SSID
  Wi-Fi band:         $BAND_LABEL
  Dashboard:          http://10.42.0.1:7000

Activating the access point disconnects the current Wi-Fi connection.
The existing connection profile is preserved so it can be restored later.
EOF


if [[ "$NM_BAND" == "a" ]]; then
  echo "Note: 5 GHz AP operation depends on the board's Wi-Fi capabilities and regulatory domain."
fi

if [[ ! -t 0 ]]; then
  echo "Run this script from an interactive local, USB/ADB, or Ethernet terminal." >&2
  exit 4
fi

read -r -p "Continue and switch $WIFI_INTERFACE to access-point mode? [y/N] " REPLY
if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
  echo "No network changes were made."
  exit 0
fi

sudo -v

if ! nmcli -t -f NAME connection show | grep -Fxq "$CONNECTION_NAME"; then
  sudo nmcli connection add \
    type wifi \
    ifname "$WIFI_INTERFACE" \
    con-name "$CONNECTION_NAME" \
    ssid "$SSID"
fi

sudo nmcli connection modify "$CONNECTION_NAME" \
  connection.interface-name "$WIFI_INTERFACE" \
  connection.autoconnect yes \
  connection.autoconnect-priority 100 \
  802-11-wireless.ssid "$SSID" \
  802-11-wireless.mode ap \
  802-11-wireless.band "$NM_BAND" \
  802-11-wireless-security.key-mgmt wpa-psk \
  802-11-wireless-security.psk "$PASSWORD" \
  ipv4.method shared \
  ipv4.addresses "$AP_ADDRESS" \
  ipv6.method disabled

sudo nmcli connection up "$CONNECTION_NAME" ifname "$WIFI_INTERFACE"

cat <<EOF

ROSPug access point is active.
  Wi-Fi:    $SSID
  Band:     $BAND_LABEL
  Password: $PASSWORD
  Web UI:   http://10.42.0.1:7000

If the dashboard is not available, confirm that the ROSPug App is running.
EOF
