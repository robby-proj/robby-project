#!/usr/bin/env python3
"""Read, validate, and publish LD19 LiDAR sector snapshots over TCP."""

from __future__ import annotations

import ctypes
import errno
import glob
import json
import os
import select
import socket
import struct
import termios
import threading
import time
import tty
from collections import deque


PORT = 8766
HEADER = 0x54
VER_LEN = 0x2C
FRAME_SIZE = 47
POINTS_PER_FRAME = 12
FRONT_OFFSET = float(os.getenv("ROSPUG_LIDAR_FRONT_OFFSET_DEG", "0"))

lock = threading.Lock()
points = deque(maxlen=6000)
status = {
    "online": False,
    "path": "",
    "error": "LD19 CH340 USB adapter not found",
    "valid_packets": 0,
    "crc_errors": 0,
    "last_packet": 0.0,
    "speed_dps": 0,
}


# UNO Q's current 7.0 kernel image can enumerate the factory CH340 adapter
# without providing the optional ch341.ko driver. These usbfs definitions let
# the Brick configure and read that adapter directly, while retaining the
# normal ttyUSB path for images that do provide the driver.
USB_VENDOR = 0x1A86
USB_PRODUCT = 0x7523
USBDEVFS_CONTROL = 0xC0185500
USBDEVFS_BULK = 0xC0185502
USBDEVFS_CLAIMINTERFACE = 0x8004550F
USBDEVFS_RELEASEINTERFACE = 0x80045510


class UsbControlTransfer(ctypes.Structure):
    _fields_ = (
        ("request_type", ctypes.c_uint8),
        ("request", ctypes.c_uint8),
        ("value", ctypes.c_uint16),
        ("index", ctypes.c_uint16),
        ("length", ctypes.c_uint16),
        ("timeout_ms", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
    )


class UsbBulkTransfer(ctypes.Structure):
    _fields_ = (
        ("endpoint", ctypes.c_uint),
        ("length", ctypes.c_uint),
        ("timeout_ms", ctypes.c_uint),
        ("data", ctypes.c_void_p),
    )


libc = ctypes.CDLL(None, use_errno=True)
libc.ioctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p)
libc.ioctl.restype = ctypes.c_int


def usb_ioctl(descriptor: int, request: int, argument) -> int:
    result = libc.ioctl(descriptor, request, ctypes.byref(argument))
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def ch340_divisor(baud: int, version: int) -> int:
    """Return the CH340 divisor using the current Linux ch341 algorithm."""
    clock = 48_000_000
    fact = 1
    prescaler = None
    for candidate in range(3, -1, -1):
        clock_divisor = 1 << (12 - 3 * candidate - fact)
        minimum = clock / (clock_divisor * 512)
        if baud > minimum:
            prescaler = candidate
            break
    if prescaler is None:
        raise ValueError(f"unsupported CH340 baud rate: {baud}")

    clock_divisor = 1 << (12 - 3 * prescaler - fact)
    divisor = clock // (clock_divisor * baud)
    if divisor < 9 or divisor > 255:
        divisor //= 2
        clock_divisor *= 2
        fact = 0
    if divisor < 2:
        raise ValueError(f"unsupported CH340 baud rate: {baud}")

    lower_error = 16 * clock // (clock_divisor * divisor) - 16 * baud
    upper_error = 16 * baud - 16 * clock // (clock_divisor * (divisor + 1))
    if lower_error >= upper_error:
        divisor += 1
    if fact == 1 and divisor % 2 == 0:
        divisor //= 2
        fact = 0

    value = ((0x100 - divisor) << 8) | (fact << 2) | prescaler
    if version > 0x27:
        value |= 0x80
    return value


def usb_control(
    descriptor: int,
    request_type: int,
    request: int,
    value: int = 0,
    index: int = 0,
    length: int = 0,
) -> bytes:
    payload = (ctypes.c_uint8 * max(1, length))()
    transfer = UsbControlTransfer(
        request_type,
        request,
        value,
        index,
        length,
        1000,
        ctypes.cast(payload, ctypes.c_void_p),
    )
    received = usb_ioctl(descriptor, USBDEVFS_CONTROL, transfer)
    return bytes(payload[:received]) if length else b""


def find_ch340_usb():
    for device in sorted(glob.glob("/sys/bus/usb/devices/*")):
        try:
            with open(os.path.join(device, "idVendor"), encoding="ascii") as handle:
                vendor = int(handle.read().strip(), 16)
            with open(os.path.join(device, "idProduct"), encoding="ascii") as handle:
                product = int(handle.read().strip(), 16)
            if vendor != USB_VENDOR or product != USB_PRODUCT:
                continue
            with open(os.path.join(device, "busnum"), encoding="ascii") as handle:
                bus = int(handle.read().strip())
            with open(os.path.join(device, "devnum"), encoding="ascii") as handle:
                number = int(handle.read().strip())
            path = f"/dev/bus/usb/{bus:03d}/{number:03d}"
            return os.open(path, os.O_RDWR | os.O_CLOEXEC), path
        except (OSError, ValueError):
            continue

    # Fallback for containers where sysfs is hidden but usbfs is mounted.
    for path in sorted(glob.glob("/dev/bus/usb/*/*")):
        descriptor = None
        try:
            descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC)
            device = os.pread(descriptor, 18, 0)
            if len(device) >= 12:
                vendor, product = struct.unpack_from("<HH", device, 8)
                if vendor == USB_VENDOR and product == USB_PRODUCT:
                    return descriptor, path
        except OSError:
            pass
        if descriptor is not None:
            os.close(descriptor)
    return None, None


class Ch340Usb:
    """Minimal read-only CH340 transport for the LD19's 230400-baud stream."""

    def __init__(self, descriptor: int, path: str):
        self.descriptor = descriptor
        self.path = path
        self.interface = ctypes.c_uint(0)
        usb_ioctl(self.descriptor, USBDEVFS_CLAIMINTERFACE, self.interface)
        try:
            version_data = usb_control(self.descriptor, 0xC0, 0x5F, length=2)
            self.version = version_data[0] if version_data else 0x30
            usb_control(self.descriptor, 0x40, 0xA1)
            divisor = ch340_divisor(230400, self.version)
            usb_control(self.descriptor, 0x40, 0x9A, 0x1312, divisor)
            if self.version >= 0x30:
                # Enable RX/TX and select 8 data bits, no parity, one stop bit.
                usb_control(self.descriptor, 0x40, 0x9A, 0x2518, 0x00C3)
            usb_control(self.descriptor, 0x40, 0xA4, 0xFFFF, 0)
        except Exception:
            self.close()
            raise

    def read(self, size: int = 4096, timeout_ms: int = 250) -> bytes:
        payload = (ctypes.c_uint8 * size)()
        transfer = UsbBulkTransfer(
            0x82,
            size,
            timeout_ms,
            ctypes.cast(payload, ctypes.c_void_p),
        )
        try:
            received = usb_ioctl(self.descriptor, USBDEVFS_BULK, transfer)
        except OSError as exc:
            if exc.errno in (errno.ETIMEDOUT, errno.EAGAIN, errno.EINTR):
                return b""
            raise
        return bytes(payload[:received])

    def close(self) -> None:
        if self.descriptor is None:
            return
        try:
            usb_ioctl(self.descriptor, USBDEVFS_RELEASEINTERFACE, self.interface)
        except OSError:
            pass
        try:
            os.close(self.descriptor)
        except OSError:
            pass
        self.descriptor = None


def crc8(data: bytes) -> int:
    """LDROBOT CRC-8, polynomial 0x4D, initial value zero."""
    crc = 0
    for value in data:
        crc ^= value
        for _ in range(8):
            crc = ((crc << 1) ^ 0x4D) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def angle_delta(angle: float, center: float) -> float:
    return (angle - center + 180.0) % 360.0 - 180.0


def candidates():
    seen = set()
    paths = []
    paths.extend(sorted(glob.glob("/host-dev/serial/by-id/usb-1a86_USB_Serial*")))
    paths.extend(sorted(glob.glob("/host-dev/ttyUSB*")))
    for path in paths:
        real = os.path.realpath(path)
        if real not in seen:
            seen.add(real)
            yield real


def configure_serial(descriptor: int) -> None:
    tty.setraw(descriptor, termios.TCSANOW)
    attrs = termios.tcgetattr(descriptor)
    speed = getattr(termios, "B230400", None)
    if speed is None:
        raise RuntimeError("B230400 is unavailable")
    attrs[4] = speed
    attrs[5] = speed
    attrs[2] |= termios.CLOCAL | termios.CREAD
    attrs[2] &= ~getattr(termios, "CRTSCTS", 0)
    termios.tcsetattr(descriptor, termios.TCSANOW, attrs)
    termios.tcflush(descriptor, termios.TCIFLUSH)


def accept_frame(frame: bytes) -> None:
    speed, start_angle = struct.unpack_from("<HH", frame, 2)
    end_angle, _ = struct.unpack_from("<HH", frame, 42)
    span = (end_angle - start_angle) % 36000
    step = span / (POINTS_PER_FRAME - 1)
    stamp = time.monotonic()
    parsed = []
    for index in range(POINTS_PER_FRAME):
        distance_mm, intensity = struct.unpack_from("<HB", frame, 6 + index * 3)
        clockwise = (start_angle + step * index) / 100.0
        counterclockwise = (-clockwise + FRONT_OFFSET) % 360.0
        parsed.append((stamp, counterclockwise, distance_mm / 1000.0, intensity))
    with lock:
        points.extend(parsed)
        status["valid_packets"] += 1
        status["last_packet"] = stamp
        status["speed_dps"] = speed
        status["online"] = True
        status["error"] = ""


def consume(buffer: bytearray) -> None:
    marker = bytes((HEADER, VER_LEN))
    while True:
        index = buffer.find(marker)
        if index < 0:
            buffer[:] = buffer[-1:] if buffer and buffer[-1] == HEADER else b""
            return
        if index:
            del buffer[:index]
        if len(buffer) < FRAME_SIZE:
            return
        frame = bytes(buffer[:FRAME_SIZE])
        if crc8(frame[:-1]) == frame[-1]:
            del buffer[:FRAME_SIZE]
            accept_frame(frame)
        else:
            with lock:
                status["crc_errors"] += 1
            del buffer[0]


def device_loop() -> None:
    buffer = bytearray()
    while True:
        path = next(candidates(), None)
        usb_descriptor = None
        usb_path = None
        if not path:
            usb_descriptor, usb_path = find_ch340_usb()
        if not path and usb_descriptor is None:
            with lock:
                status["online"] = False
                status["path"] = ""
                status["error"] = "LD19 CH340 USB adapter not found"
            time.sleep(1.0)
            continue

        descriptor = None
        usb = None
        try:
            if path:
                descriptor = os.open(path, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
                configure_serial(descriptor)
                display_path = path.replace("/host-dev", "/dev", 1)
                transport = "kernel serial"
            else:
                usb = Ch340Usb(usb_descriptor, usb_path)
                usb_descriptor = None
                display_path = f"usb:{USB_VENDOR:04x}:{USB_PRODUCT:04x}@{usb_path}"
                transport = "direct USB"
            with lock:
                status["path"] = display_path
                status["error"] = f"Waiting for valid LD19 packets ({transport})"
            buffer.clear()
            while True:
                if descriptor is not None:
                    ready, _, exceptional = select.select([descriptor], [], [descriptor], 0.25)
                    if exceptional:
                        raise OSError("LD19 serial exception")
                    if not ready:
                        continue
                    data = os.read(descriptor, 4096)
                    if not data:
                        raise OSError("LD19 disconnected")
                else:
                    data = usb.read()
                    if not data:
                        continue
                buffer.extend(data)
                consume(buffer)
        except Exception as exc:
            with lock:
                status["online"] = False
                status["error"] = f"LD19 error: {exc}"
        finally:
            if usb_descriptor is not None:
                try:
                    os.close(usb_descriptor)
                except OSError:
                    pass
            if usb is not None:
                usb.close()
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        time.sleep(0.75)


def sector_locked(center: float, width: float, now: float):
    values = [
        distance
        for stamp, angle, distance, intensity in points
        if now - stamp <= 0.40
        and intensity > 0
        and 0.05 <= distance <= 12.0
        and abs(angle_delta(angle, center)) <= width / 2.0
    ]
    if not values:
        return None, 0
    values.sort()
    return values[min(1, len(values) - 1)], len(values)


def snapshot() -> dict:
    now = time.monotonic()
    with lock:
        last_packet = status["last_packet"]
        age_ms = int((now - last_packet) * 1000) if last_packet else -1
        fresh = bool(status["online"] and 0 <= age_ms <= 500)
        # Keep one recent return per 2-degree bin. This caps the Web UI payload
        # at 180 points while preserving a responsive 360-degree local scan.
        scan_bins = {}
        for stamp, angle, distance, intensity in reversed(points):
            if now - stamp > 0.40:
                break
            if intensity <= 0 or not 0.05 <= distance <= 12.0:
                continue
            bin_index = int(((angle + 1.0) % 360.0) // 2.0)
            if bin_index not in scan_bins:
                scan_bins[bin_index] = [
                    round(angle, 1), round(distance, 3), int(intensity)
                ]
        scan = [scan_bins[index] for index in sorted(scan_bins)]
        sectors = {}
        for name, center, width in (
            ("front", 0.0, 50.0),
            ("front_left", 45.0, 70.0),
            ("front_right", -45.0, 70.0),
            ("left", 90.0, 70.0),
            ("right", -90.0, 70.0),
            ("rear", 180.0, 60.0),
            ("all", 0.0, 360.0),
        ):
            distance, count = sector_locked(center, width, now)
            sectors[name] = distance
            sectors[f"{name}_points"] = count
        return {
            "online": bool(status["online"]),
            "fresh": fresh,
            "path": status["path"],
            "baud": 230400,
            "error": status["error"],
            "valid_packets": status["valid_packets"],
            "crc_errors": status["crc_errors"],
            "last_packet_age_ms": age_ms,
            "speed_dps": status["speed_dps"],
            "front_offset_deg": FRONT_OFFSET,
            "scan": scan,
            "scan_points": len(scan),
            "sectors": sectors,
        }


def serve(connection: socket.socket) -> None:
    while True:
        payload = json.dumps(
            {"protocol": 1, "lidar": snapshot()}, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        connection.sendall(payload)
        time.sleep(0.10)


def main():
    threading.Thread(target=device_loop, name="ld19-device", daemon=True).start()
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", PORT))
    server.listen(4)
    print("[lidar-reader] listening on 8766", flush=True)
    while True:
        client, address = server.accept()
        print("[lidar-reader] client", address, flush=True)
        try:
            serve(client)
        except Exception as exc:
            print("[lidar-reader] client ended:", exc, flush=True)
        finally:
            client.close()


if __name__ == "__main__":
    main()
