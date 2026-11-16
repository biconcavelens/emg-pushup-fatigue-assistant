"""Learned model: predict the fatigue point (the rep where the set enters the
'stop zone', i.e. 20 % tempo loss) from the first K reps seen by the EMG tracker.

Features use only what the EMG-only tracker produces (rep intervals and peak
amplitudes). Trained on SIMULATED sets (see train_fatigue_model.py).
"""
from pathlib import Path

import joblib
import numpy as np

K = 8
MODEL_PATH = Path(__file__).parent / "models" / "fatigue_point.joblib"
FEATURES = ["ref_interval_ms", "interval_drift_pct_per_rep", "amp_drift_pct_per_rep",
            "interval_cv_pct", "mean_peak", "late_vs_early_interval"]


def early_features(reps):
    r = reps[:K]
    iv = np.array([x.interval_ms for x in r[1:]], float)
    pk = np.array([x.peak for x in r], float)
    t_ref = float(np.median(iv[:3]))
    drift_iv = np.polyfit(np.arange(len(iv)), iv, 1)[0] / t_ref * 100
    drift_pk = np.polyfit(np.arange(len(pk)), pk, 1)[0] / pk.mean() * 100
    cv = iv.std() / iv.mean() * 100
    return [t_ref, drift_iv, drift_pk, cv, pk.mean(), iv[-3:].mean() / t_ref - 1]


def stop_zone_rep(reps):
    """Ground-truth target: first rep at which the status reaches 'stop zone'."""
    from emg_features import fatigue_metrics
    for n in range(4, len(reps) + 1):
        if fatigue_metrics(reps[:n])["status"] in ("stop zone", "at failure"):
            return n
    return None


_model = None


def predict_stop_rep(reps):
    """Predicted rep number of the stop zone, or None if not available yet."""
    global _model
    if len(reps) < K or not MODEL_PATH.exists():
        return None
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    p = float(_model.predict([early_features(reps)])[0])
    return max(p, len(reps))
