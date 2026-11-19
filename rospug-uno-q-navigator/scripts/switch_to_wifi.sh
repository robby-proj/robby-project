#!/usr/bin/env bash
set -euo pipefail

TARGET_CONNECTION="${1:-}"
if [[ -z "$TARGET_CONNECTION" ]]; then
  echo "Usage: $0 <existing-NetworkManager-connection>" >&2
  echo "Known connections:" >&2
  nmcli -f NAME,TYPE connection show >&2
  exit 2
fi

if ! nmcli -t -f NAME connection show | grep -Fxq "$TARGET_CONNECTION"; then
  echo "Connection profile '$TARGET_CONNECTION' does not exist." >&2
  nmcli -f NAME,TYPE connection show >&2
  exit 3
fi

sudo nmcli connection modify ROSPug-AP connection.autoconnect no 2>/dev/null || true
sudo nmcli connection down ROSPug-AP 2>/dev/null || true
sudo nmcli connection up "$TARGET_CONNECTION"

echo "Connected using NetworkManager profile: $TARGET_CONNECTION"
