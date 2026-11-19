#!/usr/bin/env python3
"""Read SHANWAN/Linux joystick events and relay framed records over TCP."""
import glob, os, select, socket, struct, time
JS=struct.Struct("IhBB"); PORT=8765

def candidates():
    seen=set()
    paths = []
    for p in sorted(glob.glob('/dev/input/by-id/*-joystick')):
        if '-event-joystick' not in p:
            paths.append(p)
    paths.extend(sorted(glob.glob('/dev/input/js*')))
    for p in paths:
        real=os.path.realpath(p)
        if not os.path.basename(real).startswith('js'):
            continue
        if real not in seen:
            seen.add(real); yield real

def serve(conn):
    while True:
        fd=None; path=None
        for p in candidates():
            try: fd=os.open(p,os.O_RDONLY|os.O_NONBLOCK); path=p; break
            except OSError: pass
        if fd is None:
            conn.sendall(b'STATUS\twaiting\n'); time.sleep(1); continue
        conn.sendall(('STATUS\tconnected\t'+path+'\n').encode())
        buf=bytearray()
        try:
            while True:
                ready,_,_=select.select([fd],[],[],1.0)
                if not ready:
                    conn.sendall(b'PING\n'); continue
                data=os.read(fd,4096)
                if not data: raise OSError('disconnected')
                buf.extend(data)
                while len(buf)>=JS.size:
                    raw=bytes(buf[:JS.size]); del buf[:JS.size]
                    ms,val,typ,num=JS.unpack(raw)
                    conn.sendall(f'JS\t{typ}\t{num}\t{val}\n'.encode())
        finally:
            if fd is not None: os.close(fd)

srv=socket.socket(); srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
srv.bind(('0.0.0.0',PORT)); srv.listen(4)
print('[gamepad-reader] listening on 8765',flush=True)
while True:
    conn,addr=srv.accept(); print('[gamepad-reader] client',addr,flush=True)
    try: serve(conn)
    except Exception as e: print('[gamepad-reader] client ended:',e,flush=True)
    finally: conn.close()
