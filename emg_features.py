"""EMG-only streaming rep tracker and fatigue metrics. No video anywhere.

Input: envelope samples (ADC counts from the Advancer sensor), one at a time.
Output: completed reps with timing and amplitude, and fatigue metrics.

Rep detection is peak/valley based (push-up EMG is continuous, there is often no
quiet gap between reps): a rep is a local maximum of the smoothed envelope that
is followed by a drop to <= 60 % of its height. Peaks smaller than 60 % of the
recent typical peak (e.g. the lowering phase) are ignored.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Rep:
    n: int
    t_peak: float
    peak: float         # counts above baseline
    area: float         # counts*s above baseline since the previous valley
    interval_ms: float  # since previous rep's peak (nan for rep 1)


class RepTracker:
    PRIOR_AMP = 400.0   # typical peak (counts above baseline) before any rep is seen
    MIN_PEAK = 40.0
    REL_MIN = 0.6       # accept peaks >= 60 % of recent typical peak
    DROP = 0.6          # confirm a peak once the envelope falls below 60 % of it

    def __init__(self, fs=500.0):
        self.dt = 1 / fs
        self.s = None
        self.base = None
        self.ref_amp = self.PRIOR_AMP
        self.state = "rising"
        self.maxv = 0.0
        self.t_max = 0.0
        self.valley = 0.0
        self.area = 0.0
        self.reps = []

    def update(self, t, v):
        """Feed one sample; returns a Rep when a rep has just been confirmed."""
        if self.s is None:
            self.s = self.base = float(v)
        self.s += (v - self.s) * self.dt / 0.06
        if self.s < self.base:
            self.base += (self.s - self.base) * 0.2
        else:
            self.base += (self.s - self.base) * 0.0005
        a = max(self.s - self.base, 0.0)
        self.area += a * self.dt

        done = None
        if self.state == "rising":
            if a >= self.maxv:
                self.maxv, self.t_max = a, t
            elif self.maxv >= self.MIN_PEAK and a <= self.DROP * self.maxv:
                if self.maxv >= self.REL_MIN * self.ref_amp:
                    prev = self.reps[-1].t_peak if self.reps else None
                    done = Rep(len(self.reps) + 1, self.t_max, self.maxv, self.area,
                               (self.t_max - prev) * 1000 if prev else float("nan"))
                    self.reps.append(done)
                    self.ref_amp = 0.6 * self.ref_amp + 0.4 * self.maxv
                self.state, self.valley, self.area = "falling", a, 0.0
        else:
            self.valley = min(self.valley, a)
            if a >= self.valley + max(0.25 * self.ref_amp, 30.0):
                self.state, self.maxv, self.t_max = "rising", a, t
        return done


def fatigue_metrics(reps):
    """Tempo-loss (velocity-loss proxy), EMG amplitude rise and a status label."""
    n = len(reps)
    if n < 4:
        return {"n": n, "status": "warming up", "vl_pct": 0.0, "amp_rise_pct": 0.0}
    iv = np.array([r.interval_ms for r in reps[1:]])
    t_ref = float(np.median(iv[:3]))
    t_rec = float(np.median(iv[-3:]))
    vl = 100 * (1 - t_ref / t_rec)
    pk = np.array([r.peak for r in reps])
    amp = float(100 * (pk[-3:].mean() / pk[:3].mean() - 1))
    if vl >= 35 or t_rec > 2 * t_ref:
        status = "at failure"
    elif vl >= 20:
        status = "stop zone"
    elif vl >= 10:
        status = "building fatigue"
    else:
        status = "fresh"
    return {"n": n, "status": status, "vl_pct": round(float(vl), 1),
            "amp_rise_pct": round(amp, 1), "ref_interval_ms": round(t_ref),
            "recent_interval_ms": round(t_rec)}
