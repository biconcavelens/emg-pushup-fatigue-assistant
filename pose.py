#!/usr/bin/env python3
"""Extract the elbow angle over time from a recorded video with MediaPipe.

Usage:
    python pose.py recordings/<take_folder>

Reads  <take>/video.mp4 (+ frames.csv for timestamps if present)
Writes <take>/angles.csv   columns: t, elbow_deg, vis
Best with a side-on camera so the arm is visible in 2D.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

MODEL = Path(__file__).parent / "models" / "pose_landmarker_lite.task"
# MediaPipe landmark indices: (shoulder, elbow, wrist)
SIDES = {"left": (11, 13, 15), "right": (12, 14, 16)}


def _angle(a, b, c):
    """Angle at b (degrees) between points a-b-c, each an (x, y) pair."""
    v1, v2 = a - b, c - b
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))


def frame_times(take: Path, n_frames: int, fps: float):
    f = take / "frames.csv"
    if f.exists():
        t = np.genfromtxt(f, delimiter=",", skip_header=1)[:, 1]
        if len(t) >= n_frames:
            return t[:n_frames]
    return np.arange(n_frames) / fps


def extract(take: Path, verbose=True):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not MODEL.exists():
        raise SystemExit(f"Pose model missing: {MODEL}")
    video = take / "video.mp4"
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO)
    landmarker = vision.PoseLandmarker.create_from_options(opts)

    # per-frame angle and visibility for both arms
    ang = {s: [] for s in SIDES}
    vis = {s: [] for s in SIDES}
    n = 0
    last_ms = -1
    times_hint = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if times_hint is None:
            times_hint = frame_times(take, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10**6, fps)
        t = times_hint[n] if n < len(times_hint) else n / fps
        ms = max(int(t * 1000), last_ms + 1)
        last_ms = ms
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        for s, (i_s, i_e, i_w) in SIDES.items():
            if res.pose_landmarks:
                lm = res.pose_landmarks[0]
                pts = [np.array([lm[i].x * w, lm[i].y * h]) for i in (i_s, i_e, i_w)]
                ang[s].append(_angle(*pts))
                vis[s].append(min(lm[i].visibility for i in (i_s, i_e, i_w)))
            else:
                ang[s].append(np.nan)
                vis[s].append(0.0)
        n += 1
    cap.release()
    landmarker.close()
    if n == 0:
        raise SystemExit("Video has no frames.")

    # use the arm that is better visible overall (side-on camera sees one arm)
    best = max(SIDES, key=lambda s: np.mean(vis[s]))
    t = frame_times(take, n, fps)
    elbow = np.array(ang[best])
    v = np.array(vis[best])
    out = take / "angles.csv"
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t", "elbow_deg", "vis"])
        for ti, a, vi in zip(t, elbow, v):
            wr.writerow([f"{ti:.4f}", "" if np.isnan(a) else f"{a:.2f}", f"{vi:.3f}"])
    if verbose:
        good = np.mean((v > 0.5) & ~np.isnan(elbow))
        print(f"{n} frames, tracked arm: {best}, usable frames: {100*good:.0f}% -> {out}")
        if good < 0.7:
            print("WARNING: arm often not tracked. Use a side-on view, good light, "
                  "whole body in frame.")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("take", type=Path)
    extract(ap.parse_args().take)
