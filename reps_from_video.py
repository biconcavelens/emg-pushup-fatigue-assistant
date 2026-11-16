#!/usr/bin/env python3
"""Find push-up repetitions in a recorded video WITHOUT a pose model.

Method: dense optical flow (Farneback) between consecutive frames; the vertical
motion of the moving pixels is integrated into a body-height signal; reps are the
oscillation peaks of that signal. Timestamps come from frames.csv, so timing is
accurate to about one camera frame (~55-60 ms at 17-18 fps) before sub-frame
interpolation.

Usage:  python reps_from_video.py recordings/<take_folder>
Writes: <take>/reps.csv  (rep, t_bottom_s, t_prev_bottom_s, interval_ms)
        <take>/reps_overview.png
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def vertical_signal(video: Path):
    cap = cv2.VideoCapture(str(video))
    prev = None
    dy = [0.0]
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(fr, (160, 120)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.hypot(flow[..., 0], flow[..., 1])
            thr = max(np.percentile(mag, 80), 0.05)
            m = mag > thr
            dy.append(float(flow[..., 1][m].mean()) if m.any() else 0.0)
        prev = g
    cap.release()
    return np.cumsum(dy)


def parabolic(t, y, i):
    """Sub-frame peak time around index i."""
    if i <= 0 or i >= len(y) - 1:
        return t[i]
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    off = 0.0 if abs(den) < 1e-12 else 0.5 * (a - c) / den
    return t[i] + off * (t[i + 1] - t[i - 1]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take", type=Path)
    ap.add_argument("--min-gap", type=float, default=0.9, help="min seconds between reps")
    args = ap.parse_args()
    take = args.take

    pos = vertical_signal(take / "video.mp4")
    ft = np.genfromtxt(take / "frames.csv", delimiter=",", skip_header=1)[:, 1]
    n = min(len(pos), len(ft))
    t, pos = ft[:n], pos[:n]
    fs = 1 / np.median(np.diff(t))

    # uniform grid, remove slow drift, smooth
    tg = np.arange(t[0], t[-1], 1 / fs)
    y = np.interp(tg, t, pos)
    hb, ha = butter(2, 0.25 / (fs / 2), "high")
    lb, la = butter(2, 2.5 / (fs / 2), "low")
    y = filtfilt(lb, la, filtfilt(hb, ha, y))

    # bottoms of the push-up = maxima of downward image motion (y grows downward)
    peaks, _ = find_peaks(y, distance=int(args.min_gap * fs),
                          prominence=0.35 * np.std(y))
    tb = np.array([parabolic(tg, y, i) for i in peaks])
    with open(take / "reps.csv", "w") as f:
        f.write("rep,t_bottom_s,interval_ms\n")
        for k, tk in enumerate(tb):
            iv = "" if k == 0 else f"{(tk - tb[k-1]) * 1000:.0f}"
            f.write(f"{k+1},{tk:.3f},{iv}\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(tg, y, lw=1); ax.plot(tg[peaks], y[peaks], "rv")
    ax.set_xlabel("time (s)"); ax.set_ylabel("body height signal")
    ax.set_title(f"{len(tb)} reps detected from video motion (not a pose model)")
    fig.tight_layout(); fig.savefig(take / "reps_overview.png", dpi=110)
    iv = np.diff(tb) * 1000
    print(f"{len(tb)} reps; camera ~{fs:.1f} fps (timing resolution ~{1000/fs:.0f} ms)")
    if len(iv):
        print("intervals ms:", np.round(iv).astype(int).tolist())


if __name__ == "__main__":
    main()
