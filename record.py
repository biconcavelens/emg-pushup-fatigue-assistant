#!/usr/bin/env python3
"""Record webcam video and EMG (Arduino serial) on one shared clock.

Outputs one folder per take under recordings/:
    video.mp4    webcam video
    frames.csv   frame_idx, t   (t = seconds since take start, host clock)
    emg.csv      t, adc         (t = seconds since take start, host clock)
    meta.json    label, subject, muscle, sensor placement, rates, notes

Examples:
    python record.py --label pushup --muscle triceps --duration 20
    python record.py --label pushup --muscle triceps --fake          # no hardware
    python record.py --label pushup --muscle triceps --no-camera     # EMG only
"""
import argparse
import csv
import json
import math
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2


class EmgReader(threading.Thread):
    """Reads 'micros,adc' lines from the Arduino and stamps them with host time."""

    def __init__(self, ser, t0, fake=False):
        super().__init__(daemon=True)
        self.ser, self.t0, self.fake = ser, t0, fake
        self.rows = []  # (host_t, adc)
        self.stop = threading.Event()
        self.error = None

    def run(self):
        try:
            if self.fake:
                self._run_fake()
            else:
                self._run_serial()
        except Exception as e:  # surface to main thread
            self.error = e

    def _run_fake(self):
        # ~0.4 Hz "rep" envelope + noise, 500 Hz, for testing without hardware
        import random
        period = 1 / 500
        nxt = time.perf_counter()
        while not self.stop.is_set():
            t = time.perf_counter() - self.t0
            env = 0.5 * (1 - math.cos(2 * math.pi * 0.4 * t))
            self.rows.append((t, int(60 + 600 * env + random.gauss(0, 15))))
            nxt += period
            time.sleep(max(0, nxt - time.perf_counter()))

    def _run_serial(self):
        ser = self.ser
        ser.reset_input_buffer()
        while not self.stop.is_set():
            line = ser.readline()
            t = time.perf_counter() - self.t0
            try:
                _, adc = line.decode().strip().split(",")
                self.rows.append((t, int(adc)))
            except ValueError:
                continue  # partial / garbled line
        ser.close()


class NullReader:
    """Stand-in used with --no-emg (video only)."""
    error = None
    rows = []

    def __init__(self):
        self.stop = threading.Event()

    def start(self):
        pass

    def join(self, timeout=None):
        pass


def find_port():
    from serial.tools import list_ports
    ports = [p.device for p in list_ports.comports()
             if "ACM" in p.device or "USB" in p.device]
    return ports[0] if ports else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="exercise, e.g. pushup")
    ap.add_argument("--muscle", required=True, help="muscle under the electrodes")
    ap.add_argument("--subject", default="s01")
    ap.add_argument("--notes", default="")
    ap.add_argument("--duration", type=float, default=20.0, help="seconds")
    ap.add_argument("--port", default=None, help="serial port (auto-detect if omitted)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--preview", action="store_true", help="show live window (q to stop)")
    ap.add_argument("--fake", action="store_true", help="synthetic EMG, no Arduino")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-emg", action="store_true",
                    help="video only (no Arduino); EMG can be simulated later")
    ap.add_argument("--out", default="recordings")
    args = ap.parse_args()
    if args.no_camera and args.no_emg:
        raise SystemExit("--no-camera and --no-emg together record nothing.")

    port = None
    if not args.fake and not args.no_emg:
        port = args.port or find_port()
        if port is None:
            raise SystemExit("No Arduino found. Plug it in, or use --fake.")

    # Open the serial port first so a failure aborts before anything is recorded
    ser = None
    if not args.fake and not args.no_emg:
        import serial
        try:
            ser = serial.Serial(port, args.baud, timeout=0.1)
        except serial.SerialException as e:
            raise SystemExit(f"Cannot open {port}: {e}\n"
                             f"Permission denied? Run: sudo chmod a+rw {port}\n"
                             "Also close the Arduino IDE Serial Monitor.")
        time.sleep(2.0)  # Uno resets when the port opens

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) / f"{args.label}_{args.subject}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    cap = writer = None
    if not args.no_camera:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open camera {args.camera}")
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Measure the real capture rate: cameras often deliver ~20 fps in dim light,
        # and writing at a nominal 30 would make the video play too fast.
        for _ in range(5):
            cap.read()
        ta = time.perf_counter()
        n_ok = sum(1 for _ in range(20) if cap.read()[0])
        fps_real = n_ok / max(time.perf_counter() - ta, 1e-6)
        fps_real = min(max(fps_real, 5.0), 60.0)
        print(f"Camera delivers ~{fps_real:.0f} fps (more light usually helps)")
        writer = cv2.VideoWriter(str(out / "video.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps_real, (w, h))

    # Countdown so the subject can get into position
    for i in (3, 2, 1):
        print(f"Starting in {i}...", flush=True)
        time.sleep(1)

    t0 = time.perf_counter()
    reader = NullReader() if args.no_emg else EmgReader(ser, t0, fake=args.fake)
    reader.start()
    print(f"RECORDING {args.duration:.0f}s -> {out}", flush=True)

    frames = []
    try:
        while time.perf_counter() - t0 < args.duration:
            if reader.error:
                raise reader.error
            if cap is not None:
                ok, frame = cap.read()
                t = time.perf_counter() - t0
                if ok:
                    writer.write(frame)
                    frames.append(t)
                    if args.preview:
                        cv2.imshow("record (q to stop)", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
            else:
                time.sleep(0.05)
    finally:
        reader.stop.set()
        reader.join(timeout=2)
        if cap is not None:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()

    if not args.no_emg:
        with open(out / "emg.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["t", "adc"])
            wr.writerows(reader.rows)
    if frames:
        with open(out / "frames.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["frame_idx", "t"])
            wr.writerows(enumerate(frames))

    emg_hz = len(reader.rows) / max(reader.rows[-1][0], 1e-9) if reader.rows else 0
    vid_fps = len(frames) / frames[-1] if frames else 0
    meta = dict(vars(args), port=port, emg_samples=len(reader.rows),
                emg_hz_measured=round(emg_hz, 1), video_fps_measured=round(vid_fps, 1),
                emg_type="envelope (Advancer v3: rectified+smoothed), 10-bit ADC")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    if not reader.rows and not args.no_emg:
        print("WARNING: 0 EMG samples. Is the sketch flashed to the Uno, and is the "
              "Serial Monitor closed? This take is unusable.")
    print(f"Done. EMG: {len(reader.rows)} samples (~{emg_hz:.0f} Hz), "
          f"video: {len(frames)} frames (~{vid_fps:.0f} fps)")


if __name__ == "__main__":
    main()
