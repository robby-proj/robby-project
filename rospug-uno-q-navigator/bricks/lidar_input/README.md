# LD19 LiDAR Input

Reads the CH340 USB adapter in an isolated App Lab service, validates official
47-byte LD19 packets, calculates navigation sectors, and publishes read-only
JSON snapshots to the main application over the internal network. It uses
`/dev/ttyUSB*` when the kernel driver exists and direct usbfs access otherwise.
