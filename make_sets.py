#!/usr/bin/env python3
"""Build SIMULATED EMG sets for the demo / RAG assistant.

!! All EMG here is SIMULATED. Set 1 uses rep timings measured from the real
video (reps.csv); sets 2-4 use generated timings. Every output carries
"simulated": true in its .json so the assistant can print a banner. !!

Usage: python make_sets.py recordings/<take_folder>   -> data/sim_sets/
"""
import argparse
import json
from pathlib import Path

import numpy as np

import simulate_emg as sim

FS_G = 100.0  # drive grid, Hz
OUT = Path(__file__).parent / "data" / "sim_sets"


def bump(tg, s, e, amp):
    x = np.zeros_like(tg)
    m = (tg >= s) & (tg <= e)
    x[m] = amp * np.sin(np.pi * (tg[m] - s) / (e - s)) ** 2
    return x


def drive_from_bottoms(bottoms, fatigue_gain, rng, tail=4.0, n0=None):
    b = np.asarray(bottoms)
    n = len(b)
    T = np.diff(b)
    T = np.concatenate([[T[0]], T])           # interval *into* each bottom
    t_ref = np.median(T[1:4]) if n > 4 else T.mean()
    tg = np.arange(0, b[-1] + tail, 1 / FS_G)
    d = np.zeros_like(tg)
    n0 = max(0.6 * n, 1) if n0 is None else n0   # reps over which fatigue builds
    for k in range(n):
        t_next = T[k + 1] if k + 1 < n else T[k]
        gain = 1 + fatigue_gain * (1 - np.exp(-k / n0))
        gain *= 1 + 0.25 * np.clip(T[k] / t_ref - 1, 0, 1.5)   # slow grinding rep
        amp = gain * (1 + 0.06 * rng.standard_normal())
        d += bump(tg, b[k] - 0.45 * T[k], b[k], 0.45 * amp)            # lowering
        d += bump(tg, b[k], b[k] + 0.55 * t_next, amp)                 # pushing up
    return tg, d


def gen_bottoms(n, t0, slow, p, rng, start=2.5):
    k = np.arange(n)
    T = t0 * (1 + (slow - 1) * (k / (n - 1)) ** p) * (1 + 0.05 * rng.standard_normal(n))
    return start + np.cumsum(T)


def write_set(name, bottoms, fatigue_gain, seed, meta_extra):
    rng = np.random.default_rng(seed)
    tg, drive = drive_from_bottoms(bottoms, fatigue_gain, rng)
    t_out, adc = sim.sensor_model(tg, drive, 450, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.csv", "w") as f:
        f.write("t,adc\n")
        for ti, ai in zip(t_out, adc):
            f.write(f"{ti:.4f},{ai}\n")
    meta = {"simulated": True, "name": name, "reps_true": int(len(bottoms)),
            "bottoms_s": [round(float(x), 3) for x in bottoms],
            "fatigue_gain": fatigue_gain, **meta_extra}
    (OUT / f"{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"{name}: {len(bottoms)} reps, {t_out[-1]:.0f} s -> {OUT / (name + '.csv')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take", type=Path)
    args = ap.parse_args()
    rows = np.genfromtxt(args.take / "reps.csv", delimiter=",", skip_header=1)
    real_bottoms = rows[:, 1]

    write_set("set1_real_timing", real_bottoms, 0.9, 1,
              {"description": "EMG SIMULATED; rep timing measured from real video "
                              "(to failure, 32 reps)"})
    rng = np.random.default_rng(11)
    write_set("set2_to_failure", gen_bottoms(28, 1.25, 3.4, 2.2, rng), 0.9, 2,
              {"description": "EMG and timing SIMULATED; to failure, second set"})
    write_set("set3_fatigued", gen_bottoms(22, 1.30, 3.0, 2.0, rng), 1.0, 3,
              {"description": "EMG and timing SIMULATED; third set, fewer reps"})
    write_set("set4_stop_at_20pct", gen_bottoms(16, 1.25, 1.3, 1.5, rng), 0.45, 4,
              {"description": "EMG and timing SIMULATED; stopped near 20% tempo loss"})


if __name__ == "__main__":
    main()
