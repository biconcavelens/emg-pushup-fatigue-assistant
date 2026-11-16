#!/usr/bin/env python3
"""Live EMG bar in the terminal, for placing electrodes and checking gain.

    python live.py            # Ctrl+C to stop

Relaxed should sit low (about 10-30). A hard flex should rise clearly but stay
under 1023. Move/replace electrodes while watching the bar.
"""
import sys
import time

import serial
from serial.tools import list_ports

ports = [p.device for p in list_ports.comports() if "ACM" in p.device or "USB" in p.device]
if not ports:
    sys.exit("No Arduino found.")
try:
    ser = serial.Serial(ports[0], 115200, timeout=0.1)
except serial.SerialException as e:
    sys.exit(f"{e}\nTry: sudo chmod a+rw {ports[0]}  (and close the Serial Monitor)")
time.sleep(2)
ser.reset_input_buffer()

buf, last = [], time.time()
peak = 0
try:
    while True:
        line = ser.readline()
        try:
            buf.append(int(line.decode().strip().split(",")[1]))
        except (ValueError, IndexError):
            continue
        if time.time() - last >= 0.1 and buf:
            v = max(buf)
            peak = max(peak, v)
            bar = "#" * int(v / 1023 * 60)
            flag = "  CLIPPING" if v >= 1023 else ""
            print(f"\r{v:4d} peak {peak:4d} |{bar:<60}|{flag}   ", end="", flush=True)
            buf, last = [], time.time()
except KeyboardInterrupt:
    print()
