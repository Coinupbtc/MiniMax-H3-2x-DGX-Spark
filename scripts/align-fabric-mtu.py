#!/usr/bin/env python3
"""Set netdev MTU via ioctl (works from a NET_ADMIN host-net container)."""
from __future__ import annotations

import fcntl
import socket
import struct
import sys

SIOCSIFMTU = 0x8922


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: align-fabric-mtu.py IFACE MTU", file=sys.stderr)
        return 2
    iface = sys.argv[1]
    mtu = int(sys.argv[2])
    path = f"/sys/class/net/{iface}/mtu"
    before = open(path, encoding="utf-8").read().strip()
    ifr = struct.pack("16sI", iface.encode(), mtu) + b"\x00" * 12
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        fcntl.ioctl(sock, SIOCSIFMTU, ifr)
    finally:
        sock.close()
    after = open(path, encoding="utf-8").read().strip()
    print(f"{iface}: {before} -> {after}")
    return 0 if after == str(mtu) else 1


if __name__ == "__main__":
    raise SystemExit(main())
