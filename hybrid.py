#!/usr/bin/env python3
"""Combine the measured EMG with a pose-derived SIMULATED envelope, clearly flagged.

Per-sample labels from the measured sensor value v (ADC counts):
    v <= REST_MAX                 -> "rest"      (measured)
    REST_MAX < v < ACTIVE_MAX     -> "active"    (measured, weak signal)
    otherwise                     -> "artifact"  (measured value untrusted)

adc_sim is a separate column: the envelope simulated from the video elbow angle
(see simulate_emg.py). It is NEVER written into adc_measured. Use `source` to
see which column to trust for each sample.

Usage:
    python hybrid.py recordings/<take_folder>
Writes  <take>/emg_hybrid.csv  t, adc_measured, label, adc_sim
        <take>/meta_hybrid.json, <take>/hybrid_overview.png
"""
import argparse
import json
from pathlib import Path

import numpy as np

import simulate_emg as sim

REST_MAX = 11
ACTIVE_MAX = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take", type=Path)
    ap.add_argument("--peak", type=float, default=450)
    ap.add_argument("--fatigue", type=float, default=0.8)
    args = ap.parse_args()
    take = args.take

    d = np.genfromtxt(take / "emg.csv", delimiter=",", skip_header=1)
    t, v = d[:, 0], d[:, 1]
    label = np.where(v <= REST_MAX, "rest",
                     np.where(v < ACTIVE_MAX, "active", "artifact"))

    if not (take / "angles.csv").exists():
        import pose
        pose.extract(take)
    ta, ang = sim.load_angles(take)
    tg, a, vel = sim.process_angle(ta, ang)
    drive = sim.add_fatigue(sim.activation(a, vel), args.fatigue)
    rng = np.random.default_rng(0)
    t_sim, adc_sim = sim.sensor_model(tg, drive, args.peak, rng)
    sim_on_t = np.interp(t, t_sim, adc_sim, left=np.nan, right=np.nan)

    with open(take / "emg_hybrid.csv", "w") as f:
        f.write("t,adc_measured,label,adc_sim\n")
        for ti, vi, li, si in zip(t, v.astype(int), label, sim_on_t):
            f.write(f"{ti:.4f},{vi},{li},{'' if np.isnan(si) else int(si)}\n")

    counts = {k: int((label == k).sum()) for k in ("rest", "active", "artifact")}
    meta = {
        "thresholds": {"rest_max": REST_MAX, "active_max": ACTIVE_MAX},
        "counts": counts,
        "columns": {
            "adc_measured": "raw sensor reading, unmodified",
            "label": "rest/active/artifact decided from adc_measured only",
            "adc_sim": "SIMULATED envelope from video pose, not measured",
        },
        "note": "adc_sim must be reported as simulated; it is never mixed into adc_measured.",
    }
    (take / "meta_hybrid.json").write_text(json.dumps(meta, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"rest": "tab:gray", "active": "tab:blue", "artifact": "tab:red"}
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    for k, c in colors.items():
        m = label == k
        ax[0].plot(t[m], v[m], ".", ms=2, color=c, label=f"{k} ({m.sum()})")
    ax[0].set_ylabel("MEASURED sensor (ADC)"); ax[0].legend(loc="upper right")
    ax[1].plot(t, sim_on_t, lw=0.8, color="tab:orange")
    ax[1].set_ylabel("SIMULATED from pose"); ax[1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(take / "hybrid_overview.png", dpi=110)
    print(f"labels: {counts}; wrote emg_hybrid.csv, meta_hybrid.json, hybrid_overview.png")


if __name__ == "__main__":
    main()
