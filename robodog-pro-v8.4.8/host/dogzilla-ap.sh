#!/bin/sh
set -eu
PROFILE="dogzilla-ap"
SSID="DOGZILLA-UNOQ"
PASSWORD="dogzilla123"
IFACE="wlan0"
ADDRESS="10.42.0.1/24"
ACTION="${1:-status}"

need_nmcli(){ command -v nmcli >/dev/null 2>&1 || { echo "ERROR: nmcli not installed on UNO Q host" >&2; exit 1; }; }
install_profile(){
  if nmcli -t -f NAME connection show | grep -Fxq "$PROFILE"; then
    echo "Profile $PROFILE already exists; applying production settings."
  else
    sudo nmcli connection add type wifi ifname "$IFACE" con-name "$PROFILE" autoconnect no ssid "$SSID"
  fi
  sudo nmcli connection modify "$PROFILE" \
    802-11-wireless.mode ap \
    802-11-wireless.band bg \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk "$PASSWORD" \
    ipv4.method shared \
    ipv4.addresses "$ADDRESS" \
    ipv6.method disabled \
    connection.autoconnect no
  echo "Configured $SSID at http://10.42.0.1:7000"
}
need_nmcli
case "$ACTION" in
  install) install_profile ;;
  up) install_profile; sudo nmcli connection up "$PROFILE" ;;
  down) sudo nmcli connection down "$PROFILE" || true ;;
  restart) sudo nmcli connection down "$PROFILE" || true; sleep 1; install_profile; sudo nmcli connection up "$PROFILE" ;;
  status) nmcli -f NAME,TYPE,DEVICE,STATE connection show --active; echo; ip -4 addr show dev "$IFACE" || true ;;
  *) echo "Usage: $0 {install|up|down|restart|status}" >&2; exit 2 ;;
esac
