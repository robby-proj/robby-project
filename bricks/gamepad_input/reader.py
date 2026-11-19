#!/usr/bin/env python3
"""Read Linux joystick events and relay framed records over TCP."""

import glob
import os
import select
import socket
import struct
import time


JS = struct.Struct("IhBB")
PORT = 8765


def candidates():
    seen = set()
    paths = []
    for path in sorted(glob.glob("/dev/input/by-id/*-joystick")):
        if "-event-joystick" not in path:
            paths.append(path)
    paths.extend(sorted(glob.glob("/dev/input/js*")))
    for path in paths:
        real = os.path.realpath(path)
        if not os.path.basename(real).startswith("js"):
            continue
        if real not in seen:
            seen.add(real)
            yield real


def serve(connection):
    while True:
        descriptor = None
        path = None
        for candidate in candidates():
            try:
                descriptor = os.open(candidate, os.O_RDONLY | os.O_NONBLOCK)
                path = candidate
                break
            except OSError:
                pass
        if descriptor is None:
            connection.sendall(b"STATUS\twaiting\n")
            time.sleep(1)
            continue

        connection.sendall(("STATUS\tconnected\t" + path + "\n").encode())
        buffer = bytearray()
        try:
            while True:
                ready, _, _ = select.select([descriptor], [], [], 1.0)
                if not ready:
                    connection.sendall(b"PING\n")
                    continue
                data = os.read(descriptor, 4096)
                if not data:
                    raise OSError("gamepad disconnected")
                buffer.extend(data)
                while len(buffer) >= JS.size:
                    raw = bytes(buffer[: JS.size])
                    del buffer[: JS.size]
                    _, value, event_type, number = JS.unpack(raw)
                    connection.sendall(f"JS\t{event_type}\t{number}\t{value}\n".encode())
        finally:
            os.close(descriptor)


def main():
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", PORT))
    server.listen(4)
    print("[gamepad-reader] listening on 8765", flush=True)
    while True:
        client, address = server.accept()
        print("[gamepad-reader] client", address, flush=True)
        try:
            serve(client)
        except Exception as exc:
            print("[gamepad-reader] client ended:", exc, flush=True)
        finally:
            client.close()


if __name__ == "__main__":
    main()
