#!/usr/bin/env python3
"""
Transparent IPv6 [::1]:8790 to IPv4 127.0.0.1:8790 proxy.
Resolves macOS Safari/Chrome 'localhost' IPv6 loopback connection refusal.
"""
import socket
import select
import threading
import sys

def forward(src, dst):
    try:
        while True:
            r, _, _ = select.select([src, dst], [], [])
            if src in r:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
            if dst in r:
                data = dst.recv(65536)
                if not data:
                    break
                src.sendall(data)
    except Exception:
        pass
    finally:
        try:
            src.close()
        except Exception:
            pass
        try:
            dst.close()
        except Exception:
            pass

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8790
    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(('::1', port))
        s.listen(128)
    except OSError as e:
        print(f"[IPv6 Proxy] Could not bind ::1:{port}: {e}")
        return

    print(f"[IPv6 Proxy] Forwarding [::1]:{port} -> 127.0.0.1:{port}...")
    while True:
        try:
            client, _ = s.accept()
            target = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            target.connect(('127.0.0.1', port))
            t = threading.Thread(target=forward, args=(client, target), daemon=True)
            t.start()
        except Exception:
            pass

if __name__ == '__main__':
    main()
