#!/usr/bin/env python3
"""Simulate a triceps EMG *envelope* from the video-derived elbow angle.

!! THIS IS SIMULATED DATA, NOT A MEASUREMENT. !!
The output is written to emg_sim.csv (never emg.csv) and meta_sim.json carries
"simulated": true. A model trained on it only learns the rules coded below, so it
cannot validate any physiological claim (e.g. about fatigue). Label it as
simulated in any report or demo.

Usage:
    python simulate_emg.py recordings/<take_folder> [--peak 450] [--fatigue 0.8]

Reads  <take>/angles.csv (runs pose.py first if missing)
Writes <take>/emg_sim.csv  (t, adc @ 500 Hz, same scale as the real sensor)
       <take>/meta_sim.json, <take>/sim_overview.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, filtfilt, find_peaks

FS_OUT = 500.0   # Hz, matches the real recorder
FS_ANG = 100.0   # Hz, internal grid for angle processing
BASELINE = 11    # ADC counts, resting level observed on the real sensor


def load_angles(take: Path):
    d = np.genfromtxt(take / "angles.csv", delimiter=",", skip_header=1,
                      filling_values=np.nan)
    t, ang, vis = d[:, 0], d[:, 1], d[:, 2]
    ok = ~np.isnan(ang) & (vis > 0.4)
    if ok.sum() < 20:
        raise SystemExit("Too few tracked frames in angles.csv; re-record side-on.")
    return t[ok], ang[ok]


def process_angle(t, ang):
    """Resample to a uniform grid and low-pass; return grid time, angle, velocity."""
    tg = np.arange(t[0], t[-1], 1 / FS_ANG)
    a = np.interp(tg, t, ang)
    b, c = butter(2, 3.0 / (FS_ANG / 2))
    a = filtfilt(b, c, a)
    v = np.gradient(a, 1 / FS_ANG)  # deg/s, positive = arm extending (pushing up)
    return tg, a, v


def count_reps(tg, a):
    top, bot = np.percentile(a, 95), np.percentile(a, 5)
    peaks, _ = find_peaks(-a, prominence=0.3 * (top - bot), distance=int(0.6 * FS_ANG))
    return peaks


def activation(a, v):
    """Triceps drive 0..~1 from elbow flexion and movement phase."""
    top, bot = np.percentile(a, 95), np.percentile(a, 5)
    flex = np.clip((top - a) / max(top - bot, 1e-6), 0, 1)
    # normalise speed against the local peak speed (~3 s window) so that slower
    # late reps are not simply scaled down; fatigue is modelled separately below
    win = int(3.0 * FS_ANG)
    vref = np.maximum(maximum_filter1d(np.abs(v), size=win), 1e-6)
    v_up = np.clip(v / vref, 0, 1)      # concentric (pushing up): strongest
    v_dn = np.clip(-v / vref, 0, 1)     # eccentric (lowering): moderate
    return flex * (0.30 + 0.70 * v_up + 0.20 * v_dn)


def add_fatigue(act, k):
    """Same movement needs more drive as work accumulates (simulated fatigue)."""
    work = np.cumsum(act) / FS_ANG
    w0 = max(work[-1], 1e-6)
    return act * (1 + k * (1 - np.exp(-work / w0)))


def sensor_model(tg, drive, peak, rng):
    """Envelope sensor: fast attack, slower decay, multiplicative + additive noise."""
    t_out = np.arange(tg[0], tg[-1], 1 / FS_OUT)
    x = np.interp(t_out, tg, drive)
    # slow multiplicative variability of the EMG envelope
    noise = rng.standard_normal(len(x))
    b, c = butter(2, 8.0 / (FS_OUT / 2))
    x = x * (1 + 0.10 * filtfilt(b, c, noise) / noise.std() * 0.5)
    y = np.empty_like(x)
    dt = 1 / FS_OUT
    att, dec = np.exp(-dt / 0.03), np.exp(-dt / 0.12)
    prev = 0.0
    for i, xi in enumerate(x):
        a = att if xi > prev else dec
        prev = a * prev + (1 - a) * xi
        y[i] = prev
    adc = BASELINE + peak * y + rng.normal(0, 0.4, len(y))
    return t_out, np.clip(np.round(adc), 0, 1023).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take", type=Path)
    ap.add_argument("--peak", type=float, default=450,
                    help="ADC counts for a fresh, full-effort rep (default 450)")
    ap.add_argument("--fatigue", type=float, default=0.8,
                    help="extra drive at the end of the set, 0 = none (default 0.8)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    take = args.take

    if not (take / "angles.csv").exists():
        import pose
        pose.extract(take)
    t, ang = load_angles(take)
    tg, a, v = process_angle(t, ang)
    reps = count_reps(tg, a)
    drive = add_fatigue(activation(a, v), args.fatigue)
    rng = np.random.default_rng(args.seed)
    t_out, adc = sensor_model(tg, drive, args.peak, rng)

    with open(take / "emg_sim.csv", "w") as f:
        f.write("t,adc\n")
        for ti, ai in zip(t_out, adc):
            f.write(f"{ti:.4f},{ai}\n")
    meta = {"simulated": True, "source": "elbow angle from video via pose.py",
            "peak": args.peak, "fatigue": args.fatigue, "seed": args.seed,
            "reps_detected": int(len(reps)), "samples": int(len(adc)),
            "note": "NOT measured EMG. Do not use for physiological claims."}
    (take / "meta_sim.json").write_text(json.dumps(meta, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    ax[0].plot(tg, a, lw=1)
    ax[0].plot(tg[reps], a[reps], "rv", label=f"{len(reps)} reps (bottoms)")
    ax[0].set_ylabel("elbow angle (deg)"); ax[0].legend(loc="upper right")
    ax[1].plot(t_out, adc, lw=0.8, color="tab:green")
    ax[1].set_ylabel("SIMULATED EMG (ADC)"); ax[1].set_xlabel("time (s)")
    ax[1].set_title("simulated, not measured", fontsize=9, color="gray")
    fig.tight_layout(); fig.savefig(take / "sim_overview.png", dpi=110)
    print(f"{len(reps)} reps detected; wrote emg_sim.csv, meta_sim.json, sim_overview.png in {take}")
    print(f"simulated ADC range {adc.min()}..{adc.max()}")


if __name__ == "__main__":
    main()
